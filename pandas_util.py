from dataclasses import dataclass
import pandas as pd
import numpy as np


# get nltk and corpus

# get scapy and corpus
# Currently broken on 3.9 ??
# import swifter
# import pandas_profiling
from arrow import now
import arrow


def toPercentNP(x):
    return np.multiply(x, 100)


def toPercentForMonkeyPatch(appliable: any):
    # I'm not sure why but using switfer is *way* slower.
    # appliable.swifter.apply(toPercentNP)
    print(type(appliable))
    return appliable.swifter.apply(toPercentNP)


# Monkey Patch some methods
pd.core.series.Series.toPercent = toPercentNP  # toPercentForMonkeyPatch


@dataclass
class Measure_Helper:
    message: str
    start_time: arrow.Arrow

    def stop(self):
        print(f"-- [{(now() - self.start_time).seconds}s]: {self.message}")


def time_it(message):
    print(f"++ {message}")
    return Measure_Helper(message, now())


print("pandas util 0.02")
