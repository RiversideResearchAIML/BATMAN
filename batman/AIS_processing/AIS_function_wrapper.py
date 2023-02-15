import AIS_legality
import os
# common wrapperper
# overwrite
# multiple executions with lazy return
# df inputs
import pandas as pd
import import_AIS

def wrapper( the_func, arg = None, overwrite = False, nrows = None, multicolumn = False ):
    def wrapper_helper( x, overwrite=False):
        filename = the_func(x, return_filename=True)
        if os.path.exists(filename) and not overwrite:
            return pd.read_feather(filename)
        df = the_func(x)
        if filename:
            df.to_feather(filename)
        return df
    try:
        for a in arg:
            df = wrapper_helper( a, overwrite = overwrite, nrows = nrows )
    except:
        df = wrapper_helper( arg, overwrite = overwrite, nrows = nrows )
    if multicolumn:
        df.columns = pd.MultiIndex.from_product([['test'], df.columns])
    yield df

##
if __name__ == '__main__':
    def the_func(x, return_filename=False):
        filename = 'filename %f.feather' % x
        if return_filename:
            return filename
        return pd.DataFrame([x ** 2], columns=[''])

    def do_it(arg, overwrite=False):
        return wrapper(the_func, arg=arg, overwrite=overwrite)

    for a in do_it( range(13), overwrite = True ):
        print(a)
