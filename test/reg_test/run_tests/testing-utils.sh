setTestDirs() {

  # Determine some locations
  DIR_NAME=$(cd `dirname "${BASH_SOURCE[0]}"` && pwd -P)

  # Determine the location of the tests results (for yellowstone)
  RESULT_DIR=$(cd "${DIR_NAME}/../../results/yellowstone" && pwd -P)

  # Set the location of the "build" base directory
  if [ -n "$1" -a -d "$1" ]; then
    # Use this location as the base of the file structure for the tests
    BUILD_DIR=$1
  else
    # Set the build directory from the set file structure
    BUILD_DIR=$(cd `dirname $DIR_NAME/../../../../..` && pwd -P)/build
  fi

}

# Strip the end of the stdout file which contains lsf dump
stripAppendage() {
  sed -i -e '/^ exiting/,/^ number of MPI/{/^ exiting/!{/^ number of MPI/!d}}' -e '/^process/d' $1
}

createLSFHeader() {

  RUN_SCRIPT=$1

  #delete the file if it exists
  rm -f $RUN_SCRIPT

  # Set up some yellowstone boiler plate
  echo "#!/bin/bash" >> $RUN_SCRIPT
  echo ""  >> $RUN_SCRIPT # newlines

  echo "#BSUB -a poe" >> $RUN_SCRIPT

  # To do: move this check up and properly handle the error status
  if [ -n "$HOMME_PROJID" ]; then
    echo "#BSUB -P $HOMME_PROJID" >> $RUN_SCRIPT
  else
    echo "PROJECT CHARGE ID (HOMME_PROJID) not set"
    exit -1
  fi 

  echo "" >> $RUN_SCRIPT # newline

  echo "#BSUB -q small" >> $RUN_SCRIPT
  echo "#BSUB -W 0:20" >> $RUN_SCRIPT

  echo "" >> $RUN_SCRIPT # newline

  echo "#BSUB -x" >> $RUN_SCRIPT

  echo "" >> $RUN_SCRIPT # newline

  echo "#BSUB -R \"select[scratch_ok > 0 ]\"" >> $RUN_SCRIPT

  echo "" >> $RUN_SCRIPT # newline

  # Set the job name
  echo "#BSUB -J $testName" >> $RUN_SCRIPT
  echo "" >> $RUN_SCRIPT

  # Set the output and error filenames
  echo "#BSUB -o $testName.stdout.%J" >> $RUN_SCRIPT
  echo "#BSUB -e $testName.stderr.%J" >> $RUN_SCRIPT
  echo "" >> $RUN_SCRIPT

  # Set the ncpus and ranks per MPI
  echo "#BSUB -n $num_cpus" >> $RUN_SCRIPT
  echo '#BSUB -R "span[ptile='$num_cpus']" ' >> $RUN_SCRIPT

  echo "" >> $RUN_SCRIPT

  echo "cd $outputDir" >> $RUN_SCRIPT

  echo "" >> $RUN_SCRIPT

}


createStdHeader() {

  RUN_SCRIPT=$1

  echo "#!/bin/bash " >> $RUN_SCRIPT

  echo "" >> $RUN_SCRIPT

  echo "cd $outputDir" >> $RUN_SCRIPT

  echo "" >> $RUN_SCRIPT

}

yellowstoneExec() {
  RUN_SCRIPT=$1
  EXEC=$2
  echo "mpirun.lsf $EXEC" >> $RUN_SCRIPT

}

printSubmissionSummary() {

  # Output a summary of the test name along with the bsub job id for easy reference
  echo "" # newline
  echo "############################################################################"
  echo "Summary of submissions"
  echo "############################################################################"
  for i in $(seq 0 $(( ${#SUBMIT_TEST[@]} - 1)))
  do
    echo "Test ${SUBMIT_TEST[$i]} has ID ${SUBMIT_JOB_ID[i]}"
  done
  echo "############################################################################"

}

queueWait() {

  for i in $(seq 0 $(( ${#SUBMIT_TEST[@]} - 1)))
  do
    echo -n "Examining status of ${SUBMIT_TEST[$i]}..."
    jobID=${SUBMIT_JOB_ID[i]}
    jobFinished=false

    while ! $jobFinished;
    do
      # Test if the job exists
      jobStat=`bjobs -a $jobID | tail -n 1 | awk '{print $3}'`

      # Print the status of the job
      echo -n "...$jobStat..."

      # if the job is registered in the queue and the status is PEND or RUN then wait
      if [ -n "$jobStat" -a "$jobStat" == "PEND" -o "$jobStat" == "RUN" ]; then
        # Job still in queue or running
        sleep 20 # sleep for 20s
      else # if jobStat=DONE, EXIT or it is finished and no longer registered in the queue
        jobFinished=true
        echo "FINISHED..."
      fi
    done
  done
  echo "############################################################################"

}

#diffStdOut() {
#
#  # Should be a unique file
#  diffFile="diff.${SUBMIT_JOB_ID[0]}"
#  echo "Concatenating all diff output into $diffFile"
#
#  # Then diff with the stored results (yellowstone only)
#  for i in $(seq 0 $(( ${#SUBMIT_TEST[@]} - 1)))
#  do
#    THIS_TEST=${SUBMIT_TEST[$i]}
#
#    # Need to remove "-run" from the test name
#    #   This is an ugly hack but otherwise this takes a lot of reformatting
#    THIS_TEST=`echo $THIS_TEST | sed 's/-run//'`
#
#    # The following is not very clean
#    NEW_RESULT=${HOMME_TESTING_DIR}/${THIS_TEST}.stdout.${SUBMIT_JOB_ID[$i]}
#    SAVED_RESULT=${HOMME_TEST_RESULTS}/${THIS_TEST}/${THIS_TEST}.stdout
#
#    # TODO: make sure these files exist
#    if [ -f $NEW_RESULT ]; then
#      stripAppendage $NEW_RESULT
#      echo "diff $NEW_RESULT $SAVED_RESULT" >> $diffFile
#      # append the output to 
#      diff $NEW_RESULT $SAVED_RESULT >> $diffFile
#    else
#      echo "Result $NEW_RESULT does not exist. Perhaps job ${SUBMIT_JOB_ID[$i]} crashed or was killed"
#    fi
#
#  done
#}

submitTestsToLSF() {

  echo "Submitting ${num_submissions} jobs to queue"

  SUBMIT_TEST=()
  SUBMIT_JOB_ID=()
  SUBMIT_TEST=()

  # Loop through all of the tests
  for subNum in $(seq 1 ${num_submissions})
  do

    subFile=subFile${subNum}
    subFile=${!subFile}
    #echo "subFile=${subFile}"
    subJobName=`basename ${subFile} .sh`
    #echo "subJobName=$subJobName"

    # setup file for stdout and stderr redirection
    THIS_STDOUT=${subJobName}.out
    THIS_STDERR=${subJobName}.err

    # Run the command
    # For some reason bsub must not be part of a string
    echo -n "Submitting test ${subJobName} to the queue... "
    bsub < ${subFile} > $THIS_STDOUT 2> $THIS_STDERR
    BSUB_STAT=$?

    # Do some error checking
    if [ $BSUB_STAT == 0 ]; then
      # the command was succesful
      BSUB_ID=`cat $THIS_STDOUT | awk '{print $2}' | sed  's/<//' | sed -e 's/>//'`
      echo "successful job id = $BSUB_ID"
      SUBMIT_TEST+=( "${subJobName}" )
      SUBMIT_JOB_ID+=( "$BSUB_ID" )
    else 
      echo "failed with message:"
      cat $THIS_STDERR
      exit -1
    fi
    rm $THIS_STDOUT
    rm $THIS_STDERR
  done
}

runTestsStd() {

  echo "Submitting ${num_submissions} jobs"

  SUBMIT_TEST=()
  SUBMIT_JOB_ID=()
  SUBMIT_TEST=()

  # Loop through all of the tests
  for subNum in $(seq 1 ${num_submissions})
  do

    subFile=subFile${subNum}
    subFile=${!subFile}
    #echo "subFile=${subFile}"
    subJobName=`basename ${subFile} .sh`
    #echo "subJobName=$subJobName"

    # setup file for stdout and stderr redirection
    THIS_STDOUT=${subJobName}.out
    THIS_STDERR=${subJobName}.err

    # Run the command
    # For some reason bsub must not be part of a string
    echo -n "Running test ${subJobName} ... "
    #echo "${subFile} > $THIS_STDOUT 2> $THIS_STDERR"
    chmod u+x ${subFile}
    ${subFile} > $THIS_STDOUT 2> $THIS_STDERR &
    RUN_PID=$!
    echo "PID=$RUN_PID"
    wait $RUN_PID
    RUN_STAT=$?
    # Technically the PID is incorrect but it really doesn't matter
    RUN_PID=$!
    # Do some error checking
    if [ $RUN_STAT = 0 ]; then
      # the command was succesful
      echo "test ${subJobName} was run successfully"
      SUBMIT_TEST+=( "${subJobName}" )
      SUBMIT_JOB_ID+=( "$RUN_PID" )
    else 
      echo "failed with message:"
      cat $THIS_STDERR
      exit -1
    fi
    rm $THIS_STDOUT
    rm $THIS_STDERR
  done
}

createScripts() {
  touch $lsfListFile
  echo "num_submissions=$num_test_files" > $lsfListFile

  for testFileNum in $(seq 1 $num_test_files)
  do

    testFile=test_file${testFileNum}
    source ${!testFile}

    testName=`basename ${!testFile} .sh`

    echo "Test $testName has $num_tests pure MPI tests"
    if [ -n "$omp_num_tests" ]; then
      echo "  and $omp_num_tests Hybrid MPI + OpenMP tests"
    fi

    # Create the run script
    thisRunScript=`dirname ${!testFile}`/$testName-run.sh

    outputDir=`dirname ${!testFile}`

    # Set up header
    #yellowstoneLSFFile $thisRunScript
    submissionHeader $thisRunScript

    for testNum in $(seq 1 $num_tests)
    do
      testExec=test${testNum}
      echo "# Pure MPI test ${testNum}" >> $thisRunScript
      #echo "mpiexec -n $num_cpus ${!testExec} > $testName.out 2> $testName.err" >> $thisRunScript
      #yellowstoneExec $thisRunScript "${!testExec}"
      execLine $thisRunScript "${!testExec}" $num_cpus
      echo "" >> $thisRunScript # new line
    done

    if [ -n "$omp_num_tests" -a "${RUN_OPENMP}" == true ]; then
      echo "export OMP_NUM_THREADS=$omp_number_threads" >> $thisRunScript
      echo "" >> $thisRunScript # new line
      for testNum in $(seq 1 $omp_num_tests)
      do
         testExec=omp_test${testNum}
         echo "# Hybrid test ${testNum}" >> $thisRunScript
         #echo "mpiexec -n $omp_num_mpi ${!testExec} > $testName.out 2> $testName.err" >> $thisRunScript
         #yellowstoneExec $thisRunScript "${!testExec}"
         execLine $thisRunScript "${!testExec}" $omp_num_mpi
         echo "" >> $thisRunScript # new line
      done
    fi

    echo "subFile$testFileNum=$thisRunScript" >>  $lsfListFile

    # Reset the variables (in case they are not redefined in the next iteration)
    unset omp_num_tests
    unset num_tests

  done

}


submissionHeader() {
  RUN_SCRIPT=$1

  if [ "$HOMME_Submission_Type" = lsf ]; then
    createLSFHeader $RUN_SCRIPT
  else
    createStdHeader $RUN_SCRIPT
  fi

}

execLine() {
  RUN_SCRIPT=$1
  EXEC=$2
  NUM_CPUS=$3

  if [ "$HOMME_Submission_Type" = lsf ]; then
    echo "mpirun.lsf $EXEC" >> $RUN_SCRIPT
  else
    echo "mpiexec -n $NUM_CPUS $EXEC" >> $RUN_SCRIPT
  fi
}

diffCprnc() {

  echo "Diffing the netcdf files using cprnc"

  if [ ! -f "${CPRNC_BINARY}" ] ; then
    echo "netcdf differencing tool cprnc not found"
    exit -1
  fi

  # source the test.sh file to get the name of the nc_output_files
  source ${HOMME_TESTING_DIR}/${TEST_NAME}/${TEST_NAME}.sh

  # nc_output_files is defined in the .sh file
  FILES="${nc_output_files}"

  if [ -z "{$FILES}" ] ; then
    echo "Test ${TEST_NAME} doesn't have Netcdf output files"
  fi

  # for files in movies
  for file in $FILES 
  do
    echo "file = ${file}"
    baseFilename=`basename $file`

    # new result
    newFile=${HOMME_TESTING_DIR}/${TEST_NAME}/movies/$file
    if [ ! -f "${newFile}" ] ; then
      echo "ERROR: The result file ${newFile} does not exist exiting" 
      exit -1
    fi

    # result in the repo
    repoFile=${HOMME_NC_RESULTS_DIR}/${TEST_NAME}/${baseFilename}
    if [ ! -f "${newFile}" ] ; then
      echo "ERROR: The repo file ${repoFile} does not exist exiting" 
      exit -1
    fi

    cmd="${CPRNC_BINARY} ${newFile} ${repoFile}"

    diffStdout=${TEST_NAME}.${baseFilename}.out
    diffStderr=${TEST_NAME}.${baseFilename}.err

    echo "Running cprnc:"
    echo "  $cmd"
    $cmd > $diffStdout 2> $diffStderr

    # Parse the output file to determine if they were identical
    DIFF_RESULT=`grep -e 'diff_test' $diffStdout | awk '{ print $8 }'`

    if [ "${DIFF_RESULT}" == IDENTICAL ] ; then
      echo "The files are identical: DIFF_RESULT=${DIFF_RESULT}"
      # Delete the output file to remove clutter
      rm $diffStdout
      rm $diffStderr
    else
      echo "The files are different: DIFF_RESULT=${DIFF_RESULT}"
      echo "  The diff output is available in $diffStdout"
      exit -1
    fi

    
  done
}

diffStdout() {

  PARSE_RESULT=${HOMME_TESTING_DIR}/${TEST_NAME}.stdout
  SAVED_RESULT=${HOMME_TEST_RESULTS}/${TEST_NAME}/${TEST_NAME}.stdout

  cmd="${PYTHON_EXECUTABLE} diffTol.py --maxTol=1.e-12 ${PARSE_RESULT} ${SAVED_RESULT}"

  echo $cmd

  diffOutput=$( $cmd )
  diffCode=$?

  if [ "$diffCode" == 0 ] ; then
    # parse output to get status
    fileDifference=`echo $diffOutput | awk  '{print $3}'`

    if [ "${fileDifference}" == identical ]; then
      echo "${diffOutput}"
      exit 0
    elif [ "${fileDifference}" == similar ]; then 
      echo "${diffOutput}"
      exit 0
    elif [ "${fileDifference}" == different ]; then
      echo "${diffOutput}"
      exit -1
    else
      echo "File comparison failed to yield expected result (identical,simlar,different)"
      echo "diffTol Output: ${diffOutput}"
      exit -1
    fi

  else
    echo "$cmd failed with error code $diffCode"
    echo "diffTol Output: ${diffOutput}"
    exit -1
  fi

}

parseStdout() {

  # Then diff with the stored results (yellowstone only)
  for i in $(seq 0 $(( ${#SUBMIT_TEST[@]} - 1)))
  do
    THIS_TEST=${SUBMIT_TEST[$i]}

    # Need to remove "-run" from the test name
    #   This is an ugly hack but otherwise this takes a lot of reformatting
    THIS_TEST=`echo $THIS_TEST | sed 's/-run//'`

    # The following is not very clean
    NEW_RESULT=${HOMME_TESTING_DIR}/${THIS_TEST}.stdout.${SUBMIT_JOB_ID[$i]}
    PARSE_RESULT=${HOMME_TESTING_DIR}/${THIS_TEST}.stdout

    if [ "$HOMME_Submission_Type" = lsf ]; then
      stripAppendage $NEW_RESULT
    fi
    grep -e '=' ${NEW_RESULT} | grep -iv bsub | grep -ive 't_init' > ${PARSE_RESULT}

  done
}
