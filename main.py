# read and check datasets
from classifier import OneTimeTest
from compare_clfs.CompareClassifiers import CompareClassifiers
from datetime import datetime

from tests.TStudentTest import TStudentTest

time_start = datetime.now()

repeats = [1, 2, 3, 4, 5, 10, 20, 25, 30, 40, 50, 60]
repeat = [1]

# porównanie algorytmów
#CompareClassifiers().compare()

# jedno powtórzenie
# OneTimeTest.RunBoostingIB(repeats=repeat).run()

# więcej powtórzeń
OneTimeTest.RunBoostingIB(repeats=repeats).run()

# t-student
TStudentTest().test()

time_end = datetime.now()

print(f'Czas startu: {time_start}\nCzas zakonczenia: {time_end}')


