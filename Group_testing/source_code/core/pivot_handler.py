import pandas as pd
import dateutil
import datetime

class PeriodHandler:
    def period(self, df, test_arg, period_type, periods_start, period_arg):
        if period_type == 'time':

            if (len(periods_start)==2) and (test_arg=='One-Sample'):
                raise Exception(f'Two-Samples test needs 1 period of time. Two periods were found')

            if (len(periods_start)==1) and (test_arg=='Two-Samples'):
                raise Exception(f'Two-Samples test needs 2 periods of time. One period was found')
            
            if len(period_arg)==2:
                months = dateutil.relativedelta.relativedelta(months=period_arg[0],days=period_arg[1])
            elif len(period_arg)==1:
                months = dateutil.relativedelta.relativedelta(months=period_arg[0])

            return [ df[(df.index >= periods_start[i]) & (df.index < periods_start[i]+months)] for i in range( len(periods_start) ) ]
            
        elif period_type == 'point':

            periods_start = periods_start[0]

            if ( periods_start ==2 ) and (test_arg=='One-Sample'):
                raise Exception(f'Two-Samples test needs 1 period of time. Two periods were found')

            if ( periods_start ==1) and (test_arg=='Two-Samples'):
                raise Exception(f'Two-Samples test needs 2 periods of time. One period was found')

            ret = []

            for i in range(periods_start):
                df.sample(frac=1)
                ret.append( df.head(period_arg[0]) )
                df = df.iloc[period_arg[0]:,:]

            return ret

        else :
            raise Exception(f'Undefined period segmentation type.\n{period_type} not found.')