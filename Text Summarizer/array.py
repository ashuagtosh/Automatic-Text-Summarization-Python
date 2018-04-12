import numpy as np
import pandas as pd
def main():
    lst1 = range(10,20)
    lst2 = range(10)
    lst3 = range(10)
    percentile_list = pd.DataFrame(
    {'lst1Tite': lst1,
     'lst2Tite': lst2,
     'lst3Tite': lst3
    })
    print(percentile_list['lst1Tite'][2])
    return
main()
