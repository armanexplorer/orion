#include "scheduler.h"
#include <cctype>
#include <cstring>
#define MIN_DURATION 100000 // might need to change this - emperical

using namespace std;

// globals
void* klib;
vector<vector<op_info>> op_info_vector;
vector<string> client_model_names;
int* fidx;
int* num_client_kernels;
int* num_client_max_iters;
int* num_client_cur_iters;
bool* locked;

std::chrono::time_point<std::chrono::high_resolution_clock>* client_starts;
std::chrono::time_point<std::chrono::high_resolution_clock>* total_client_starts;
bool** client_starts_set;
vector<vector<float>> client_durations;

int max_sms = 132; // h100
queue<struct func_record>** client_buffers;
pthread_mutex_t** client_mutexes;
queue<struct func_record>** buffers;
int* seen;
cudnnHandle_t* global_handle0;
cudnnHandle_t* global_handle1;

// fifo-globals
cudaStream_t sched_stream;
cudaEvent_t sched_event;

// profile-globals
cudaStream_t** sched_streams;
cudaEvent_t*** events;
int* streams;
int* event_ids;
int status;
vector<int> max_sms_clients;
vector<bool> is_train;

// reef
int lp_idx = 0;
int penalty = 0;
bool** request_status;
bool* stops;
bool* stop_ack;


void Scheduler::profile_reset(int num_clients) {

	for (int i=0; i<num_clients; i++) {
		seen[i] = 0;
		streams[i] = -1;
		fidx[i] = 0;
	}
}

void Scheduler::profile_prep(queue<func_record>** qbuffers, int num_clients, bool reef) {

	register_functions();
	client_buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		client_buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	int num = num_clients;

	sched_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));
	for (int i=0; i<num; i++)
		sched_streams[i] = NULL;

	events = (cudaEvent_t***)malloc((num)*sizeof(cudaEvent_t**));
	for (int i=0; i<num; i++)
		events[i] = NULL;

	create_streams(sched_streams, num, reef);
	create_events(events, num);

	for (int i=0; i<num_clients; i++)
		printf("Stream for client %d is %d\n", i, *(sched_streams[i]));

	seen = (int*)calloc(num,sizeof(int));
	event_ids = (int*)calloc(num, sizeof(int));

	streams = (int*)malloc(num_clients*sizeof(int));
	for (int i=0; i<num_clients; i++)
		streams[i] = -1;

	sched_stream = 0;

	status = -1;

}


void Scheduler::schedule_reef(vector<func_record*> frecords, int num_clients, int depth) {

	// schedule based on REEF policy

	if (num_clients==1) {
		if (frecords[0] != NULL) {
			schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
			pop_from_queue(client_buffers[0], client_mutexes[0], 0);
		}
		return;
	}

	int hp_client = num_clients-1;
	// check for malloc operations
	for (int i=0; i<num_clients; i++) {
		if (frecords[i] != NULL) {
			if (frecords[i]->type == MALLOC_RECORD || num_client_cur_iters[i] <= 10 || num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
				schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
				pop_from_queue(client_buffers[i], client_mutexes[i], i);
				return;
			}
		}
	}

	// if hp is found, schedule
	if (frecords[hp_client] != NULL) {
		int hp_idx = seen[hp_client];

		schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
		pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);

		op_info op_info_1 = op_info_vector[hp_client][hp_idx];

		// check all kernels, and find suitable
		for (int i=0; i<hp_client; i++) {
			if (frecords[i] != NULL) {
				op_info op_info_0 = op_info_vector[i][seen[i]];
				if (op_info_0.duration <= op_info_1.duration && op_info_0.sm_used >= op_info_1.sm_used) {
					// colocate
					DEBUG_PRINT("SCHEDULE seen[0]=%d\n", seen[0]);
					schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
					pop_from_queue(client_buffers[i], client_mutexes[i], i);
					// if one is found, exit
					return;
				}
			}
		}
	}
	else {
		for (int i=0; i<hp_client; i++) {
			if (frecords[i] != NULL)
				penalty += 1;
		}
		if (penalty>=depth) {
			// schedule all
			for (int i=0; i<hp_client; i++) {
				if (frecords[i] != NULL) {

					schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
					pop_from_queue(client_buffers[i], client_mutexes[i], i);
				}
			}
			penalty = 0;
		}
	}
}

int Scheduler::schedule_sequential(vector<func_record*> frecords, int num_clients, int start) {

	// schedule based on temporal sharing

	// 1 client
	if (num_clients==1) {
		if (frecords[0] != NULL) {
			schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
			pop_from_queue(client_buffers[0], client_mutexes[0], 0);
		}
		return start;
	}


	int hp_client = num_clients-1;
	// check high priority first
	if (frecords[hp_client] != NULL) {
		bool schedule = true;
		if ((num_client_cur_iters[hp_client] <= 10) || (frecords[hp_client]->type == MALLOC_RECORD))  {
			schedule = true;
		}
		else {
			// check that no other client is "active"
			for (int i=0; i<hp_client; i++) {
				if (seen[i]>0) {
					// another client is served
					schedule=false;
					break;
				}
			}
 		}
		if (schedule) {
			// schedule on stream 0
			schedule_kernel(*(frecords[hp_client]), sched_streams[0], hp_client, events[0][event_ids[0]], seen, event_ids, 0);
			pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
		}
	}


	int end = start + num_clients; // start+1+num_clients-1
	for (int t=start+1; t<end; t++) {
		int i = t%(num_clients-1);
		if (frecords[i] != NULL) {
			bool schedule = true;
			if (num_client_cur_iters[i] <= 10 || (frecords[i]->type == MALLOC_RECORD))
				schedule = true;
			else {
				// check non-else is running
				for (int j=0; j<num_clients; j++) {
					if (i!=j && seen[j]>0) {
						// another client is served
						schedule=false;
						break;
					}
				}
			}
			if (schedule)  {
				// schedule on stream 0
				schedule_kernel(*(frecords[i]), sched_streams[0], i, events[0][event_ids[0]], seen, event_ids, 0);
				pop_from_queue(client_buffers[i], client_mutexes[i], i);
				start = t;
				break;
			}
		}
	}

	return start;

}

void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
	int start0 = 0;
	int start1 = 0;

	int prev_large = -1;
	int hp_running = -1;

	bool inf_finished = false;
	bool started = false;
 	std::chrono::time_point<std::chrono::system_clock> start_time;
	auto start_total = std::chrono::high_resolution_clock::now();

	vector<bool> total_client_set(num_clients, false);
	vector<int> profiles(num_clients, -1);
	vector<int> cur_sms(num_clients, -1);
	int hp_client = num_clients-1;

	bool large_found = false;
	long sum = 0; // sum of durations of ongoing BE kernels
	long size = 0; // sum of sizes of in-the-queues BE kernels
	int start = -1;

	// BS - works only for 2 clients for now
	// TODO: check this
	int low_sms = 0;
	int high_sms = max_sms_clients[0]; // 0 is the lp client
	int sm_threshold = max_sms_clients[0]/2;
	float hp_iter_duration = 0.0; // 1 is the hp client
	float hp_limit_float = (float)hp_limit;

	// if hp is inference, use max_sms + also there is no update phase
	if (!is_train[hp_client]) {
		sm_threshold = max_sms;
		update_start = INT_MAX;
	}

	// Optional: auto-detect the true per-iteration record count (num_kernels)
	// for workloads that have placeholder/mismatched configs or slight
	// per-iteration variability.
	//
	// Motivation: if configured num_kernels is too small, the scheduler may
	// "finish" an iteration early and exit a warmup phase, while the client is
	// still issuing CUDA work. Since the interposer blocks kernel launches until
	// the scheduler drains the queue, this causes a deadlock.
	// Conversely, if configured num_kernels is too large (or the workload issues
	// slightly fewer records on some iterations), the scheduler can wait forever
	// for a record that will never arrive, while the client is blocked inside
	// backend_lib.block(it).
	//
	// This fallback is env-gated and only applies to YOLOv5 clients.
	// It detects an end-of-iteration by observing a sustained idle gap with an
	// empty client queue after at least one record has been scheduled.
	auto env_flag_enabled = [](const char* name) -> bool {
		const char* v = getenv(name);
		return (v && v[0] != '\0' && strcmp(v, "0") != 0);
	};

	const char* auto_env = getenv("ORION_AUTO_NUM_KERNELS_SEC");
	double auto_sec = 0.0;
	if (auto_env) {
		auto_sec = atof(auto_env);
	}
	const bool auto_enabled = (auto_sec > 0.0);
	const bool auto_log = env_flag_enabled("ORION_LOG_AUTO_NUM_KERNELS") || env_flag_enabled("ORION_LOG_KERNEL_COUNTS");
	vector<std::chrono::time_point<std::chrono::high_resolution_clock>> last_progress;
	vector<bool> auto_yolo;
	const int AUTO_PLACEHOLDER_KERNELS = INT_MAX / 4;
	// Guard against false positives: there can be brief CPU-side gaps between
	// intercepted CUDA calls even mid-iteration. Only infer an iteration boundary
	// after we've seen a substantial number of scheduled records.
	const int AUTO_MIN_SEEN = 200;
	if (auto_enabled) {
		last_progress.resize(num_clients, std::chrono::high_resolution_clock::now());
		auto_yolo.resize(num_clients, false);
		for (int i = 0; i < num_clients; i++) {
			if (i >= (int)client_model_names.size()) {
				continue;
			}
			std::string m = client_model_names[i];
			for (auto &c : m) c = (char)tolower(c);
			if (m.find("yolov5") == std::string::npos) {
				continue;
			}
			auto_yolo[i] = true;
			// Use a large placeholder so we never stop early; we'll infer the true
			// boundary each iteration via idle-gap detection.
			num_client_kernels[i] = AUTO_PLACEHOLDER_KERNELS;
			if (auto_log) {
				printf(
					"AUTO_NUM_KERNELS enabled client=%d idle_threshold_s=%.3f placeholder=%d\n",
					i,
					auto_sec,
					AUTO_PLACEHOLDER_KERNELS
				);
			}
		}
	}

	while(1) {
		auto now = std::chrono::high_resolution_clock::now();
		vector<func_record*> frecords(num_clients, NULL);
		size = 0;

		for (int i=0; i<num_clients; i++) {
			if ((seen[i] == num_client_kernels[i]))
				continue;

			pthread_mutex_lock(client_mutexes[i]);
			volatile int sz = client_buffers[i]->size();
			if (sz > 0) {
				frecords[i] = &(client_buffers[i]->front());
				int cur_iter = num_client_cur_iters[i];
				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					client_starts[i] = std::chrono::high_resolution_clock::now();
					client_starts_set[i][cur_iter] = true;
					if (!total_client_set[i]) {
						total_client_starts[i] = std::chrono::high_resolution_clock::now();
						total_client_set[i] = true;
					}
				}
				//if (seen[i] == num_client_kernels[i]-1)
				//	continue;
			}
			pthread_mutex_unlock(client_mutexes[i]);
		}

		if (reef) {
			schedule_reef(frecords, num_clients, depth);
		}
		else if (seq) {
			start = schedule_sequential(frecords, num_clients, start);
		}
		else {
			if (frecords[hp_client] != NULL) { // high priority
				int op_idx_1 = seen[hp_client];
				if (op_idx_1 >= (int)op_info_vector[hp_client].size()) {
					op_idx_1 = (int)op_info_vector[hp_client].size() - 1;
					if (op_idx_1 < 0) op_idx_1 = 0;
				}
				op_info op_info_1 = op_info_vector[hp_client][op_idx_1];
				schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
				if (auto_enabled) {
					last_progress[hp_client] = now;
				}
				streams[hp_client] = 1;
				profiles[hp_client] = op_info_1.profile;
				cur_sms[hp_client] = op_info_1.sm_used;

				status = 1;
				pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
			}
			//start = -1;
			int end = start + num_clients; // start+1+num_clients-1
			for (int t=start+1; t<end; t++) {
				// Do round-robin for the BE clients
				int j = t % (num_clients-1);
				if (frecords[j] != NULL) { // low priority
						int op_idx_0 = seen[j];
						if (op_idx_0 >= (int)op_info_vector[j].size()) {
							op_idx_0 = (int)op_info_vector[j].size() - 1;
							if (op_idx_0 < 0) op_idx_0 = 0;
						}
						op_info op_info_0 = op_info_vector[j][op_idx_0];
					bool schedule = false;

					//printf("%d, %d, %d\n", low_sms, high_sms, sm_threshold);

					if ((num_clients==1) || (seen[hp_client]==0) || (frecords[j]->type == MALLOC_RECORD) || (frecords[j]->type == MEMCPY_RECORD) || (frecords[j]->type == MEMSET_RECORD) || (frecords[j]->type == FREE_RECORD))
						schedule = true;
					else if (num_client_cur_iters[j] <= 10 || num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
						schedule = true;
					}
					else if (seen[hp_client] >= update_start && (op_info_0.sm_used <= sm_threshold && cudaEventQuery(*(events[hp_client][update_start-1])) == cudaSuccess)) // && (op_info_0.sm_used <= 10*sm_threshold))
						schedule = true;
					else if (seen[hp_client]>0 && (size + op_info_0.sm_used <= sm_threshold) &&  ((op_info_0.profile == -1 || profiles[hp_client]==-1 || (profiles[hp_client] != op_info_0.profile))))
						schedule = true;
					if (schedule && large_found) {
						bool do_schedule = true;
						for (int k=0; k<num_clients-1; k++) {
						 	if (event_ids[k]>=1) {
								cudaError_t status = cudaEventQuery(*(events[k][event_ids[k]-1]));
								if (status != cudaSuccess) {
									do_schedule = false;
									break;
								}
							}
						}
						if (do_schedule) {
							large_found = false;
							sum = 0;
						}
						else
							schedule = false;
					}
					if (schedule) {
						//if (op_info_0.duration > depth && num_client_cur_iters[1] < num_client_max_iters[1] && seen[1]==0) {
							//block = true;
						size += op_info_0.sm_used;
						if ((frecords[j]->type != MALLOC_RECORD) && (frecords[j]->type != MEMCPY_RECORD) && (frecords[j]->type != MEMSET_RECORD) && (frecords[j]->type != FREE_RECORD))
							sum += op_info_0.duration;
						if (sum > depth && num_client_cur_iters[hp_client] < num_client_max_iters[hp_client]) {
							large_found = true;
						}
						schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
						if (auto_enabled) {
							last_progress[j] = now;
						}
						status = 0;
						pop_from_queue(client_buffers[j], client_mutexes[j], j);

						streams[j] = 0;
						start = j;
					}
				}
			}
		}

		// Per-iteration auto-completion for YOLOv5: if the client queue stays empty
		// and we observe a sustained idle gap, infer the end-of-iteration boundary
		// as the number of records scheduled so far (seen[i]).
		if (auto_enabled) {
			for (int i = 0; i < num_clients; i++) {
				if (i >= (int)auto_yolo.size() || !auto_yolo[i]) {
					continue;
				}
				if (seen[i] <= 0) {
					continue;
				}

				size_t qsz = 0;
				pthread_mutex_lock(client_mutexes[i]);
				qsz = client_buffers[i]->size();
				pthread_mutex_unlock(client_mutexes[i]);

				double idle_s = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_progress[i]).count();
				if (
					qsz == 0
					&& idle_s >= auto_sec
					&& num_client_kernels[i] == AUTO_PLACEHOLDER_KERNELS
					&& seen[i] >= AUTO_MIN_SEEN
				) {
					if (auto_log) {
						printf(
							"AUTO_NUM_KERNELS client=%d infer_iter_end seen=%d idle_s=%.3f\n",
							i,
							seen[i],
							idle_s
						);
					}
					num_client_kernels[i] = seen[i];
				}
			}
		}

		int finished = 0;
		for (int i=0; i<num_clients; i++) {
			//printf("Client %d, seen is %d, all is %d\n", i, seen[i], num_client_kernels[i]);
			if (
				(num_client_cur_iters[i] == num_client_max_iters[i])
				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
				|| (stop_ack[i] == true)
			)
				finished += 1;
			else if ((seen[i] == num_client_kernels[i])) {
				// check if GPU work for this client has finished
				if (!locked[i]) {
					pthread_mutex_lock(client_mutexes[i]);
					locked[i] = true;
					DEBUG_PRINT("LOCK CLIENT %d\n", i);
				}
				bool ready = true;
				if (seq) {
					if (event_ids[0] >= 1) {
						cudaError_t status = cudaEventQuery(*(events[0][event_ids[0]-1]));
						if (status != cudaSuccess)
							ready &= false;
					}
				}
				else {
					if (event_ids[i] >= 1) {
						cudaError_t status = cudaEventQuery(*(events[i][event_ids[i]-1]));
						if (status != cudaSuccess)
							ready &= false;
					}
				}
				if (ready) {
					const char* log_counts_env = getenv("ORION_LOG_KERNEL_COUNTS");
					if (log_counts_env && atoi(log_counts_env) > 0) {
						// At this point the client mutex is held (locked[i] == true).
						// seen[i] == configured num_kernels; fidx[i] is the interposer-captured record count.
						printf(
							"KCOUNT client=%d iter=%d seen=%d configured_num_kernels=%d captured_records=%d opinfo_rows=%zu queue_size=%zu\n",
							i,
							num_client_cur_iters[i],
							seen[i],
							num_client_kernels[i],
							fidx[i],
							op_info_vector[i].size(),
							client_buffers[i]->size()
						);
					}
					// if yes, reset meta-structures for this client, and let it continue
					seen[i] = 0;
					if (seq)
						event_ids[0] = 0;
					event_ids[i] = 0;
					streams[i] = -1;
					fidx[i] = 0;
					request_status[i][num_client_cur_iters[i]] = true;
					pthread_mutex_unlock(client_mutexes[i]);
					DEBUG_PRINT("UNLOCK CLIENT %d\n", i);
					num_client_cur_iters[i] += 1;
					locked[i] = false;

					// If YOLO is in auto mode, reset back to the placeholder for the next iteration.
					if (auto_enabled && i < (int)auto_yolo.size() && auto_yolo[i]) {
						num_client_kernels[i] = AUTO_PLACEHOLDER_KERNELS;
					}

					auto end = std::chrono::high_resolution_clock::now();
					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
					duration /= 1000.0;
					client_durations[i].push_back(duration);
					if (!reef && !seq && i==hp_client && is_train[hp_client]) {
						printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
						hp_iter_duration += duration;
						if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
							float hp_avg_duration = hp_iter_duration/10.0;
							printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
							hp_iter_duration = 0;

							// TODO: add better stopping conditions
							if (hp_avg_duration > hp_limit_float) {
								high_sms = sm_threshold;
								sm_threshold = (low_sms+high_sms)/2;
							}
							else {
								low_sms = sm_threshold;
								sm_threshold = (low_sms+high_sms)/2;
							}
						}
					}
					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
				}
				if (
					(num_client_cur_iters[i] == num_client_max_iters[i])
					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
					|| (stop_ack[i] == true)
				) {
					finished += 1;
					if (!warmup) {
						auto end_total = std::chrono::high_resolution_clock::now();
						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
						duration /= 1000.0;
						printf("Client %d, Total loop took %f sec\n", i, duration);
						if (i==num_clients-1) {
							for (int k=0; k<num_clients-1; k++) {
								printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
								if (!locked[k])
									pthread_mutex_lock(client_mutexes[k]);
								stops[k] = true;
								if (!locked[k])
									pthread_mutex_unlock(client_mutexes[k]);
							}
						}
					}
				}
			}
		}

		if (finished==num_clients) {
			printf("EXIT LOOP!\n");
			break;
		}

	}
	if (!warmup) {
		auto end_total = std::chrono::high_resolution_clock::now();
		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
		duration /= 1000.0;
		printf("Total loop took %f sec\n", duration);
		//process_eval(client_durations);
	}

	return NULL;
}

extern "C" {

	Scheduler* sched_init() {

		Scheduler* sched = new Scheduler();
		return sched;
	}


	void populate_kernel_info(char* kernel_info_file, vector<op_info> &ops) {

		// TODO: make this more generic, e.g. pass files/models w.r.t input
		printf("KERNEL_INFO_FILE IS %s\n", kernel_info_file);
		string line;
		std::ifstream infile(kernel_info_file);
		assert (infile.is_open());

		// ignore header
		std::getline(infile, line);

		while (std::getline(infile, line))
		{

			vector<string> v;
			stringstream sline = stringstream(line);
			while (sline.good()) {
        		string substr;
        		getline(sline, substr, ',');
        		v.push_back(substr);
    		}

			op_info info = {v[0], stoi(v[1]), stoi(v[2]), stoi(v[3]), stof(v[4])};

			ops.push_back(info);
		}

		infile.close();

	}

	void setup_change(Scheduler* scheduler, int client_id, char* file, int num_kernels) {

		// needed for backward

		op_info_vector[client_id].clear();
		populate_kernel_info(file, op_info_vector[client_id]);
		int max_sm_used = 0;
		for (auto info: op_info_vector[client_id])
			max_sm_used = max(max_sm_used, info.sm_used);
		max_sms_clients[client_id] = max_sm_used;
		num_client_kernels[client_id] = num_kernels;

	}

	void setup(
		Scheduler* scheduler,
		int num_clients,
		int* tids,
		char** models,
		char** files,
		int* num_kernels,
		int* num_iters,
		bool* train,
		bool reef,
		bool sequential
	) {

		struct passwd *pw = getpwuid(getuid());
		const char *homedir = pw ? pw->pw_dir : nullptr;
		const char* lib_path = "/orion/src/cuda_capture/libinttemp.so";
		std::string full_lib_path;
		if (homedir) {
			full_lib_path = std::string(homedir) + lib_path;
		} else {
			full_lib_path = std::string(lib_path);
		}

		klib = dlopen(full_lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);

		if (!klib) {
			fprintf(stderr, "Error: %s\n", dlerror());
			return;
		}

#ifdef SYS_gettid
		pid_t mytid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif

		// 1. thread structures
		pid_t** thread_ids_all = (pid_t**)dlsym(klib, "thread_ids");
		*thread_ids_all = (pid_t*)malloc((2*num_clients+1)*sizeof(pid_t)); // 2*N threads + scheduler

		for (int i=0; i<num_clients; i++)
			(*thread_ids_all)[i] = tids[i];
		(*thread_ids_all)[num_clients] = mytid;
		for (int i=num_clients+1; i<2*num_clients+1; i++)
			(*thread_ids_all)[i] = 0;
		//printf("address is %p, %p\n", thread_ids_all, *thread_ids_all);

		int** num_total_clients = (int**)dlsym(klib, "num_total_clients");
		*num_total_clients = (int*)malloc(sizeof(int));
		**num_total_clients = num_clients;

		for (int i=0; i<=num_clients; i++) {
			DEBUG_PRINT("Scheduler setup the thread id at %d to be %d\n", i, (*thread_ids_all)[i]);
		}

		// 2. metadata structures
		client_model_names.clear();
		for (int i=0; i<num_clients; i++) {
			client_model_names.push_back(std::string(models[i] ? models[i] : ""));
			op_info_vector.push_back({});
			client_durations.push_back({});
			populate_kernel_info(files[i], op_info_vector[i]);
			int max_sm_used = 0;
			for (auto info: op_info_vector[i])
				max_sm_used = max(max_sm_used, info.sm_used);
			max_sms_clients.push_back(max_sm_used);
			printf("----------- SIZE: %d\n", op_info_vector[i].size());
			is_train.push_back(train[i]);
		}

		// 3. indexes
		int** fidx_ptr = (int**)dlsym(klib, "func_indexes");
		*fidx_ptr = (int*)calloc(num_clients, sizeof(int));
		fidx = *fidx_ptr;

		num_client_kernels = num_kernels;
		num_client_max_iters = num_iters;

		num_client_cur_iters = (int*)calloc(num_clients, sizeof(int));
		locked = (bool*)calloc(num_clients, sizeof(bool));

		// to get measurements
		client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		total_client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		client_starts_set = (bool**)malloc(num_clients*sizeof(bool*));
		for (int i=0; i<num_clients; i++) {
			client_starts_set[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}

		// 4. communication queues + locks
		queue<func_record>*** buffers_ptr = (queue<func_record>***)dlsym(klib, "kqueues");
		*buffers_ptr = (queue<func_record>**)malloc(num_clients*sizeof(queue<func_record>*));
		queue<func_record>** buffers = *buffers_ptr;
		for (int i=0; i<num_clients; i++) {
			buffers[i] = new queue<func_record>();
			printf("size is %d\n", buffers[i]->size());
		}

		pthread_mutex_t*** client_mutexes_ptr = (pthread_mutex_t***)dlsym(klib, "mutexes");
		*client_mutexes_ptr = (pthread_mutex_t**)malloc(num_clients*sizeof(pthread_mutex_t*));
		client_mutexes = *client_mutexes_ptr;
		for (int i=0; i<num_clients; i++) {
			client_mutexes[i] = new pthread_mutex_t(); //(pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		}
		scheduler->profile_prep(buffers, num_clients, reef);

		// 5. runtime control
		bool*** request_status_ptr = (bool***)dlsym(klib, "client_request_status");
		*request_status_ptr = (bool**)malloc(num_clients*sizeof(bool*));
		request_status = *request_status_ptr;

		bool* is_sequential = (bool*)dlsym(klib, "is_sequential");
		*is_sequential = sequential;

		cudaStream_t*** client_sched_streams_ptr = (cudaStream_t***)dlsym(klib, "client_streams");
		*client_sched_streams_ptr = sched_streams;

		// check!
		bool** stops_ptr = (bool**)dlsym(klib, "client_stop");
		*stops_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stops = *stops_ptr;

		bool** stop_ack_ptr = (bool**)dlsym(klib, "client_stop_ack");
		*stop_ack_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stop_ack = *stop_ack_ptr;

		bool** affinity_set_ptr = (bool**)dlsym(klib, "affinity_set");
		(*affinity_set_ptr) = (bool*)calloc(num_clients+1, sizeof(bool));

		for (int i=0; i<num_clients; i++) {
			request_status[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}

	}


	void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int reef_depth, int hp_limit, int update_start) {

		DEBUG_PRINT("entered sched func!\n");
		if (profile_mode)
			scheduler->busy_wait_profile(num_clients, iter, warmup, warmup_iters, reef, seq, reef_depth, hp_limit, update_start);
		DEBUG_PRINT("exited sched func!\n");
		return NULL;
	}

	void* reset(Scheduler* scheduler, int num_clients) {
		scheduler->profile_reset(num_clients);
		return NULL;
	}

	// Reset per-client iteration counters and block-status flags.
	//
	// This is useful for workloads that issue CUDA work during model load (e.g.
	// YOLOv5 via torch.hub). Orion's Python frontend runs a short warmup-setup
	// schedule before the actual warmup iterations; without a reset, that setup
	// work consumes one logical iteration (num_client_cur_iters increments),
	// causing the subsequent warmup phase to end early and potentially deadlock
	// clients that are still issuing kernel launches.
	void* reset_iters(Scheduler* scheduler, int num_clients) {
		// Reset scheduler-side per-iteration state.
		scheduler->profile_reset(num_clients);

		// Reset logical iteration counters.
		if (num_client_cur_iters) {
			for (int i = 0; i < num_clients; i++) {
				num_client_cur_iters[i] = 0;
			}
		}

		// Clear per-iteration unblock flags so clients don't skip block(it).
		if (request_status && num_client_max_iters) {
			for (int i = 0; i < num_clients; i++) {
				if (!request_status[i]) {
					continue;
				}
				for (int it = 0; it < num_client_max_iters[i]; it++) {
					request_status[i][it] = false;
				}
			}
		}

		// Best-effort: clear timing markers.
		if (client_starts_set && num_client_max_iters) {
			for (int i = 0; i < num_clients; i++) {
				if (!client_starts_set[i]) {
					continue;
				}
				for (int it = 0; it < num_client_max_iters[i]; it++) {
					client_starts_set[i][it] = false;
				}
			}
		}

		return NULL;
	}
}
