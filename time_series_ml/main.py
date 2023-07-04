from learntools.core import globals_binder as binder

# binder.Binder().bind(globals())
# from learntools.python.ex1 import *

from pathlib import Path

# from learntools.time_series.style import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

data_dir = Path("./input/ts-course-data")
comp_dir = Path("./input/store-sales-time-series-forecasting")

book_sales = pd.read_csv(
    data_dir / "book_sales.csv", index_col="Date", parse_dates=["Date"]
).drop("Paperback", axis=1)

book_sales["Time"] = np.arange(len(book_sales.index))
book_sales["Lag_1"] = book_sales["Hardcover"].shift(1)
book_sales = book_sales.reindex(columns=["Hardcover", "Time", "Lag_1"])

ar = pd.read_csv(data_dir / "ar.csv")

dtype = {
    "store_nbr": "category",
    "family": "category",
    "sales": "float32",
    "onpromotion": "uint64",
}
store_sales = pd.read_csv(
    comp_dir / "train.csv",
    dtype=dtype,
    parse_dates=["date"],
    infer_datetime_format=True,
)

store_sales = store_sales.set_index("date").to_period("D")
store_sales = store_sales.set_index(["store_nbr", "family"], append=True)
average_sales = store_sales.groupby("date").mean()["sales"]

if __name__ == "__main__":
    # time series
    fig, ax = plt.subplots()
    ax.plot("Time", "Hardcover", data=book_sales, color="0.75")
    ax = sns.regplot(
        x="Time",
        y="Hardcover",
        data=book_sales,
        ci=None,
        scatter_kws=dict(color="0.25"),
    )
    ax.set_title("Time Plot of Hardcover Book Sales")
    # plt.show()

    # lag
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax1.plot(ar["ar1"])
    ax1.set_title("Series 1")
    ax2.plot(ar["ar2"])
    ax2.set_title("Series 2")
    # plt.show()
