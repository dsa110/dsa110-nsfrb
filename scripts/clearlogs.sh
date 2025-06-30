#!/bin/bash

ls $NSFRBDIR-logfiles/* | xargs -n 1 truncate -s 0
while sleep 3600
do
	ls $NSFRBDIR-logfiles/* | xargs -n 1 truncate -s 0
done
