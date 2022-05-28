#include <iostream>
#include <string>
#include <deque>
#include <vector>
#include <sstream>
#include <csignal>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <limits>
#include <pthread.h>

#define NOERROR 0
#define OVERFLOW 1

namespace {

    int num_consumer_threads{};
    int max_consumer_sleep_time{};
    std::deque<int> data{};
    sig_atomic_t is_sigint = false;
    pthread_mutex_t mutex_data = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t get_new_number = PTHREAD_COND_INITIALIZER;
    pthread_cond_t wait_consumer = PTHREAD_COND_INITIALIZER;
    pthread_barrier_t wait_all_consumers;
    thread_local int is_overflow = NOERROR;
    bool state = false;

}

namespace details {

    void destroy() {
        pthread_mutex_destroy(&::mutex_data);
        pthread_cond_destroy(&::get_new_number);
        pthread_cond_destroy(&::wait_consumer);
        pthread_barrier_destroy(&::wait_all_consumers);
    }

    bool check_sum_edges(int lth, int rth) {
        if (rth > 0) {
            return lth > std::numeric_limits<int>::max() - rth;
        } else {
            return lth < std::numeric_limits<int>::min() - rth;
        }
    }

    class lock {
    public:
        lock(pthread_mutex_t* mutex) : mutex_(mutex) {
            pthread_mutex_lock(mutex_);
        }

        ~lock() {
            pthread_mutex_unlock(mutex_);
        }

    private:
        pthread_mutex_t* mutex_;
    };
}

int get_last_error() {
    return ::is_overflow;
}

void set_last_error(int code) {
    ::is_overflow = code;
}

void* producer_routine(void* /*arg*/) {
    pthread_barrier_wait(&::wait_all_consumers);
    std::string line;
    std::getline(std::cin, line);
    std::stringstream ssm(line);
    for (int num; ssm >> num && !::is_sigint;) {
        if (num == 0) {
            continue;
        }
        {
            details::lock lk{&::mutex_data};
            ::data.emplace_back(num);
        }
        pthread_cond_signal(&::get_new_number);
        {
            details::lock lk{&::mutex_data};
            while (!is_sigint && !::data.empty()) {
                pthread_cond_wait(&::wait_consumer, &mutex_data);
            }
        }
    }
    {
        details::lock lk{&::mutex_data};
        ::state = true;
    }
    pthread_cond_broadcast(&get_new_number);
    pthread_exit(nullptr);
}

void* consumer_routine(void* /*arg*/) {
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr);
    pthread_barrier_wait(&::wait_all_consumers);
    int sum = 0;
    {
        details::lock lk{&::mutex_data};
        while (!is_sigint && !::state) {
            while (data.empty()) {
                pthread_cond_wait(&::get_new_number, &::mutex_data);
                if (::is_sigint || ::state) {
                    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
                    pthread_exit(new int[2]{sum, get_last_error()});
                }
            }
            int num = data.front();
            data.pop_front();

            pthread_mutex_unlock(&::mutex_data);
            {
                pthread_cond_signal(&::wait_consumer);
                if (details::check_sum_edges(sum, num)) {
                    set_last_error(OVERFLOW);
                    break;
                }
                sum += num;
                usleep(random() % (::max_consumer_sleep_time + 1) * 1000);
            }
            pthread_mutex_lock(&::mutex_data);
        }
    }
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
    pthread_exit(new int[2]{sum, get_last_error()});
}

void* consumer_interruptor_routine(void* arg) {
    pthread_barrier_wait(&::wait_all_consumers);
    auto& consumer_threads = *static_cast<std::vector<pthread_t>*>(arg);
    while (!::state && !is_sigint) {
        pthread_cancel(consumer_threads[rand() % ::num_consumer_threads]);
    }
    pthread_exit(nullptr);
}

int run_threads() {
    signal(SIGINT, [](int) {
        ::is_sigint = true;
        pthread_cond_broadcast(&::wait_consumer);
        pthread_cond_broadcast(&::get_new_number);
    });

    pthread_t producer_thread;
    pthread_t interrupter_thread;
    std::vector<pthread_t> consumers_threads(::num_consumer_threads);
    {
        pthread_barrier_init(&::wait_all_consumers, nullptr, ::num_consumer_threads + 3);
        pthread_create(&producer_thread, nullptr, producer_routine, nullptr);
        pthread_create(&interrupter_thread, nullptr, consumer_interruptor_routine,
                           reinterpret_cast<void*>(&consumers_threads));
        for (auto& consumer_thread : consumers_threads) {
            pthread_create(&consumer_thread, nullptr, consumer_routine, nullptr);
        }
    }
    pthread_barrier_wait(&::wait_all_consumers);
    pthread_join(interrupter_thread, nullptr);
    pthread_join(producer_thread, nullptr);
    int sum = 0;
    for (auto& consumer_thread : consumers_threads) {
        int* result;
        if (get_last_error()) {
            pthread_join(consumer_thread, reinterpret_cast<void**>(&result));
            delete[] result;
            continue;
        }
        pthread_join(consumer_thread, reinterpret_cast<void**>(&result));
        if (result[1] == OVERFLOW || details::check_sum_edges(sum, result[0])) {
            set_last_error(OVERFLOW);
        } else {
            sum += result[0];
        }
        delete[] result;
    }
    return sum;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        details::destroy();
        return 1;
    }
    std::srand(time(nullptr));
    ::num_consumer_threads = std::stoi(argv[1]);
    ::max_consumer_sleep_time = std::stoi(argv[2]);
    if (::num_consumer_threads <= 0 || ::max_consumer_sleep_time < 0) {
        details::destroy();
        return 1;
    }
    int sum = run_threads();
    if (get_last_error()) {
        std::cout << "overflow" << std::endl;
        details::destroy();
        return 1;
    } else {
        std::cout << sum << std::endl;
    }
    details::destroy();
    return 0;
}