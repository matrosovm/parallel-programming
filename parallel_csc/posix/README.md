# posix

Ваша задача - реализовать классический паттерн producer-consumer с небольшими дополнительными условиями. Программа должна состоять из 3+N потоков:

    главный
    producer
    interruptor
    N потоков consumer

На стандартный ввод программе подается строка - список чисел, разделённых пробелом. Длина списка чисел не задаётся - считывание происходит до перевода каретки.

    Задача producer-потока - получить на вход список чисел, и по очереди использовать каждое значение из этого списка для обновления переменной разделяемой между потоками

    Задача consumer-потоков отреагировать на уведомление от producer и набирать сумму полученных значений. Также этот поток должен защититься от попыток потока-interruptor его остановить. Дополнительные условия:
        Функция, исполняющая код этого потока consumer_routine, должна принимать указатель на объект/переменную, из которого будет читать обновления
        После суммирования переменной поток должен заснуть на случайное количество миллисекунд, верхний предел будет передан на вход приложения (0 миллисекунд также должно корректно обрабатываться). Вовремя сна поток не должен мешать другим потокам consumer выполнять свои задачи, если они есть
        Потоки consumer не должны дублировать вычисления друг с другом одних и тех же значений
        Поток должен отслеживать арифметическое переполнение, если очередная операция сложения приведет к переполнению, поток должен выставить код ошибки OVERFLOW(1) и завершиться
        В качестве возвращаемого значения поток должен вернуть свою частичную посчитанную сумму и код ошибки

    Задача потока-interruptor проста: пока происходит процесс обновления значений, он должен постоянно пытаться остановить случайный поток consumer (вычисление случайного потока происходит перед каждой попыткой остановки). Как только поток producer произвел последнее обновление, этот поток завершается.

В программе должена быть предусмотрена обработка ошибок через механизм статус-кодов, устанавливаемых для каждого потока индивидуально при помощи функций set_last_error(int) и get_last_error(). Хранение текущего значения ошибки должно осуществляться в TLS-переменной. Значение NOERROR(0) означает отсутствие ошибки.

Программа должна корректно обрабатывать комбинацию Ctrl+C. При нажатии, потоки должны корректно остановиться и программа должна вывести сумму, подсчитанную на данный момент.

Функция run_threads() должна запускать все потоки, дожидаться их выполнения, и возвращать результат общего суммирования. Если возникло арифметическое переполнение, необходимо вывести overflow (в нижнем регистре) и вернуть код возврата 1 из main().

Для обеспечения межпоточного взаимодействия допускается использование только pthread API. На вход приложения передаётся 2 аргумента при старте именно в такой последовательности:

    Число потоков consumer
    Верхний предел сна consumer в миллисекундах

В поток вывода должно попадать только результирующее значение, по умолчанию никакой отладочной или запросной информации выводиться не должно.

Решение должно содержать один cpp файл, который должен компилироваться командой:
'''g++ -std=c++14 -Wall -Wextra -pedantic -pthread *.cpp -o lab.a'''