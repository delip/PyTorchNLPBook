#! /bin/bash

# For each file, add a download.py line
# Any additional processing on the downloaded file

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Yelp Reviews dataset
mkdir -p $HERE/yelp
if [ ! -f $HERE/yelp/raw_train.csv ]; then
    python download.py 1xeUnqkhuzGGzZKThzPeXe2Vf6Uu_g_xM $HERE/yelp/raw_train.csv # 12536
fi
if [ ! -f $HERE/yelp/raw_test.csv ]; then
    python download.py 1G42LXv72DrhK4QKJoFhabVL4IU6v2ZvB $HERE/yelp/raw_test.csv # 4
fi
if [ ! -f $HERE/yelp/reviews_with_splits_lite.csv ]; then
    python download.py 1Lmv4rsJiCWVs1nzs4ywA9YI-ADsTf6WB $HERE/yelp/reviews_with_splits_lite.csv # 1217
fi

# Surnames Dataset
mkdir -p $HERE/surnames
if [ ! -f $HERE/surnames/surnames.csv ]; then
    python download.py 1MBiOU5UCaGpJw2keXAqOLL8PCJg_uZaU $HERE/surnames/surnames.csv # 6
fi
if [ ! -f $HERE/surnames/surnames_with_splits.csv ]; then
    python download.py 1T1la2tYO1O7XkMRawG8VcFcvtjbxDqU- $HERE/surnames/surnames_with_splits.csv # 8
fi

# Books Dataset
mkdir -p $HERE/books
if [ ! -f $HERE/books/frankenstein.txt ]; then
    python download.py 1XvNPAjooMyt6vdxknU9VO_ySAFR6LpAP $HERE/books/frankenstein.txt # 14
fi
if [ ! -f $HERE/books/frankenstein_with_splits.csv ]; then
    python download.py 1dRi4LQSFZHy40l7ZE85fSDqb3URqh1Om $HERE/books/frankenstein_with_splits.csv # 109

fi

# AG News Dataset
mkdir -p $HERE/ag_news
if [ ! -f $HERE/ag_news/news.csv ]; then
    python download.py 1hjAZJJVyez-tjaUSwQyMBMVbW68Kgyzn $HERE/ag_news/news.csv # 188
fi
if [ ! -f $HERE/ag_news/news_with_splits.csv ]; then
    python download.py 1Z4fOgvrNhcn6pYlOxrEuxrPNxT-bLh7T $HERE/ag_news/news_with_splits.csv # 208
fi

mkdir -p $HERE/nmt
if [ ! -f $HERE/nmt/eng-fra.txt ]; then 
    python download.py 1o2ac0EliUod63sYUdpow_Dh-OqS3hF5Z $HERE/nmt/eng-fra.txt # 292
fi 
if [ ! -f $HERE/nmt/simplest_eng_fra.csv ]; then 
    python download.py 1jLx6dZllBQ3LXZkCjZ4VciMQkZUInU10 $HERE/nmt/simplest_eng_fra.csv # 30
fi 
