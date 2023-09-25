import sys
import matplotlib.pyplot as plt
import pandas as pd

def highmean_plot(start_day):
   import numpy as np
   lot_name = ['sichung', 'unam', 'ohsaek1', 'ohsaek2']
   for lot in lot_name:
    highmean = pd.read_csv(f'./Visualization/{lot}_{start_day}_finalHighMeanPrediction(Test).csv', index_col = 0)
    ytrue = pd.read_csv(f'./답지파일(그저확인용-안봐도됨)/{lot}_답지_24~30.csv', index_col = 0)
    
    y = ytrue['누적'].values
    hm = np.array(highmean).flatten()

    plt.figure(figsize = (30,10))
    plt.title(f'{lot} Prediction')
    plt.plot(highmean.index, hm, label = 'highmean')
    plt.plot(highmean.index, y, label = 'true')
    plt.legend()
    plt.savefig(f'./Visualization/{lot}_HighmeanPlot.png')
    

def all_plot(start_day):
   import numpy as np
   lot_name = ['sichung', 'unam', 'ohsaek1', 'ohsaek2']
   for lot in lot_name:
    four = pd.read_csv(f'./Visualization/{lot}_{start_day}_finalPrediction(Test).csv', index_col = 0)
    ytrue = pd.read_csv(f'./답지파일(그저확인용-안봐도됨)/{lot}_답지_24~30.csv', index_col = 0)
    y = ytrue['누적'].values

    plt.figure(figsize = (30,10))
    plt.title(f'{lot} Prediction')
    plt.plot(four.index, four['전체기간'].values, label = 'total')
    plt.plot(four.index, four['절반기간'].values, label = 'half')
    plt.plot(four.index, four['쿼터기간'].values, label = 'quarter')
    plt.plot(four.index, four['피자기간'].values, label = 'one_eight')
    plt.plot(four.index, y, label = 'true')
    plt.legend()
    plt.savefig(f'./Visualization/{lot}_fourPlot.png')
    
if __name__ == '__main__':
    start_day = str(sys.argv[1])
    highmean_plot(start_day)
    all_plot(start_day)
