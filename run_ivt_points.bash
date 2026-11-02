#!/bin/bash

# Runs ivt_points.py for each ensemble member

BASEDIR=/nfs/tornlab_rit/hs559346

MEMBERS=$(ls -d ${BASEDIR}/member[0-9]* | grep -v '_lagranto\|_new\|_anal\|run\|output' | sed 's|.*/member||')

for MEMBER in $MEMBERS; do

    echo "============================================"
    echo "Processing IVT points for member ${MEMBER}"
    echo "============================================"

    DIAGFILE=${BASEDIR}/member${MEMBER}/diag.2023-01-04_00.00.00.nc
    if [ ! -f ${DIAGFILE} ]; then
        echo "Diag file not found for member ${MEMBER}, skipping"
        continue
    fi

    python ${BASEDIR}/ivt_points.py --member ${MEMBER}

    if [ $? -eq 0 ]; then
        echo "Member ${MEMBER} IVT points completed successfully"
    else
        echo "Member ${MEMBER} IVT points FAILED"
    fi

done
