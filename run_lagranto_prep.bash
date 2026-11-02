#!/bin/bash

# runs lagranto_mpas_prep.py for each ensemble member

INIT=2023010300
FHRMAX=120
BASEDIR=/nfs/tornlab_rit/hs559346

MEMBERS=$(ls -d ${BASEDIR}/member[0-9]* | grep -v '_lagranto\|_new\|_anal\|run\|output' | sed 's|.*/member||')

for MEMBER in $MEMBERS; do

    echo "============================================"
    echo "Processing member ${MEMBER}"
    echo "============================================"

    LAGRANTO_DIR=${BASEDIR}/member${MEMBER}_lagranto
    
    # Check if latlon files already exist in the lagranto directory
    LATLON_COUNT=$(ls ${LAGRANTO_DIR}/latlon.current.*.nc 2>/dev/null | wc -l)

    if [ ${LATLON_COUNT} -ge 6 ]; then
        echo "Member ${MEMBER} already has ${LATLON_COUNT} latlon files, skipping"
        continue
    fi

    # No latlon files found - delete the folder and start fresh
    echo "No latlon files found for member ${MEMBER}, deleting and reprocessing"
    rm -rf ${LAGRANTO_DIR}
    mkdir -p ${LAGRANTO_DIR} 

    cd ${LAGRANTO_DIR}

    cp ${BASEDIR}/target_domain .

    python ${BASEDIR}/lagranto_mpas_prep.py --init ${INIT} --fhrmax ${FHRMAX} --member ${MEMBER}

    if [ $? -eq 0 ]; then
        echo "Member ${MEMBER} completed successfully"
    else
        echo "Member ${MEMBER} FAILED"
    fi

    cd ${BASEDIR}

done
