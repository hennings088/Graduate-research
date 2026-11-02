#!/bin/bash

# runs lagranto itself for each ensemble member

BASEDIR=/nfs/tornlab_rit/hs559346
LAGRANTO=${BASEDIR}/lagranto.mpas/code/prog/caltra
REF=20230103_0000

#MEMBERS=$(ls -d ${BASEDIR}/member[0-9]* | grep -v '_lagranto\|_new\|_anal\|run\|output' | sed 's|.*/member||')
MEMBERS="12 23 26 36"

for MEMBER in $MEMBERS; do

    echo "============================================"
    echo "Running LAGRANTO for member ${MEMBER}"
    echo "============================================"

    LAGRANTO_DIR=${BASEDIR}/member${MEMBER}_lagranto
    TRAJ_START=${BASEDIR}/traj_start_${MEMBER}
    OUTFILE=${LAGRANTO_DIR}/forward_member${MEMBER}_ivt_start_2023010300.nc

    # Check if traj_start file exists for this member
    if [ ! -f ${TRAJ_START} ]; then
        echo "traj_start file not found for member ${MEMBER}, skipping"
        continue
    fi

    # Check if latlon files exist
    LATLON_COUNT=$(ls ${LAGRANTO_DIR}/latlon.current.*.nc 2>/dev/null | wc -l)
    if [ ${LATLON_COUNT} -eq 0 ]; then
        echo "No latlon files found for member ${MEMBER}, skipping"
        continue
    fi

    # cd into lagranto dir so caltra can find the latlon files
    cd ${LAGRANTO_DIR}

    ${LAGRANTO} ${TRAJ_START} 60 ${OUTFILE} -ref ${REF} -o 60 -ts 10

    if [ $? -eq 0 ]; then
        echo "Member ${MEMBER} LAGRANTO completed successfully"
    else
        echo "Member ${MEMBER} LAGRANTO FAILED"
    fi

    cd ${BASEDIR}

done
