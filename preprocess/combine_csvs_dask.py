import csv
import datetime
import dask.bag as db
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-t', "--tmp", type=str, default="./tmp", help="Path to dir containing the identitiy csv files")
parser.add_argument('-c', "--csv", type=str, default="./preprocess/dataset.csv", help="Path to save the combined dataset csv file")
p = parser.parse_args()

def main():

    b = db.read_text(p.tmp + '/*.csv', blocksize=100000)  # Read in a bunch of CSV files from the current directory.
    records = b.str.strip().str.split(',')
    header = records.compute()[0]  # Get the first line from the records to retrieve the header.

    combined_bag = db.from_sequence(records.compute(), npartitions=1)
    # ^ Join the bag into one partition, so the CSV file is not separated.
    filtered_bag = combined_bag.filter(lambda r: not r[0].startswith(header[0]))
    # ^ Remove the header from each CSV.

    date_today = datetime.datetime.now()

    outfile = open(p.csv , 'wb')
    bagwriter = csv.writer(outfile)

    bagwriter.writerow(header)
    for line in filtered_bag.compute():
        bagwriter.writerow(line)

    outfile.close()

if __name__ == '__main__':
    main()