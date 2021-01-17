#!/usr/bin/env bash
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -t|--tag)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        TAG=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

if [ -z ${TAG+x} ]; then echo "--tag not provided" && exit 1; else echo "tag is set to '$TAG'"; fi

if [ ${TAG} == "no" ];
then
  docker build . -t dhonerkamp/kinematic-feasibility-rl:latest;
   echo "--tag set to no, not pushing image" && exit 1;
fi
docker build . -t dhonerkamp/kinematic-feasibility-rl:${TAG} -t dhonerkamp/kinematic-feasibility-rl:latest \
  && docker push dhonerkamp/kinematic-feasibility-rl:${TAG} \
  && docker push dhonerkamp/kinematic-feasibility-rl:latest