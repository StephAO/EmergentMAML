from itertools import zip_longest
import csv

def write_csv(file_name, data):
    """
    write data to the provided csv file file_name
    data: a dictionary where the key is the column name and the value is 
    the list of values to write
    """
    
    assert(data is not None)
    
    file_name = file_name.strip()
    file_name = filename if file_name.endswith('.csv') else file_name + '.csv'
    
    _data_iter = zip_longest(*data.values(), fillvalue='')
    
    with open(file_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        csv_writer.writerow(data.keys())
        csv_writer.writerows(_data_iter)