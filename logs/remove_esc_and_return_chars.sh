#!/bin/bash
echo "sed -i 's/\r/\n/g ; s/\x1B\[2K//g' "$1
sed -i 's/\r/\n/g ; s/\x1B\[2K//g' $1
