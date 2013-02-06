# This file is used primarily to provide macro definitions and set the default locations for dependencies on supported systems
INCLUDE(DeterminePaths)

# Determine the host name
EXECUTE_PROCESS(COMMAND uname -n
  RESULT_VARIABLE Homme_result
  OUTPUT_VARIABLE Homme_output
  ERROR_VARIABLE Homme_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

IF (Homme_result EQUAL 0 AND Homme_error STREQUAL "")
  SET(Homme_Raw_Hostname ${Homme_output})
  MESSAGE(STATUS "Raw Hostname = ${Homme_Raw_Hostname}")
ELSE ()
  MESSAGE(FATAL_ERROR "Hostname could not be determined")
ENDIF()


EXECUTE_PROCESS(COMMAND uname -s
  RESULT_VARIABLE Homme_result
  OUTPUT_VARIABLE Homme_output
  ERROR_VARIABLE Homme_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

IF (Homme_result EQUAL 0 AND Homme_error STREQUAL "")
  SET(Homme_OS ${Homme_output})
ELSE ()
  MESSAGE(FATAL_ERROR "OS type could not be determined")
ENDIF()

EXECUTE_PROCESS(COMMAND whoami
  RESULT_VARIABLE Homme_result
  OUTPUT_VARIABLE Homme_output
  ERROR_VARIABLE Homme_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

IF (Homme_result EQUAL 0 AND Homme_error STREQUAL "")
  SET(Homme_Username ${Homme_output})
  MESSAGE(STATUS "Homme_Username = ${Homme_Username}")
ELSE ()
  MESSAGE(FATAL_ERROR "Username type could not be determined")
ENDIF()

# Now parse the host name to see if it matches yslogin*
# MATCHES Does REGEX where "." is a wildcard
IF (${Homme_Raw_Hostname} MATCHES "yslogin.")
  SET(Homme_Hostname "Yellowstone")
  SET(Homme_Registered_Host TRUE)
ELSEIF (${Homme_Raw_Hostname} STREQUAL "jaguar")
  SET(Homme_Hostname "Jaguar")
  SET(Homme_Registered_Host TRUE)
ELSE ()
  SET(Homme_Hostname ${Homme_Raw_Hostname})
  SET(Homme_Registered_Host FALSE)
ENDIF()

IF (Homme_Registered_Host)
  MESSAGE(STATUS "Registered Host, reading paths from systemPaths")
  readRegistredPaths()

  MESSAGE(STATUS "Homme_NETCDF_DIR = ${Homme_NETCDF_DIR}")
  MESSAGE(STATUS "Homme_PNETCDF_DIR = ${Homme_PNETCDF_DIR}")
  MESSAGE(STATUS "Homme_Hints = ${Homme_PNETCDF_DIR}")

ELSE ()
  MESSAGE(STATUS "Host not registered setting hints ")
  determineHintPaths()
  MESSAGE(STATUS "Homme_Hint_Paths=${Homme_Hint_Paths}")
ENDIF ()
