#! /bin/bash

# compress video from mjpeg to h264 codec
compress() {

  local codec_name=$(ffprobe -v error -select_streams  v:0 -show_entries \
    stream=codec_name -of default=noprint_wrappers=1:nokey=1 \
    "$1")

  # todo handle if file is already transcoded
  if [ "$codec_name" = "mjpeg" ]; then
    local filename=${1%.*}
    local extension=${1##*.}
    local outfile=${filename}_h265.${extension}

    echo "$1 is a mjpeg, transcoding..."

    # vanilla option, older codec for better compatibility
    #ffmpeg -i "$1" -c:v libx264 -crf 19 -preset fast -vsync 2 "$outfile"

    # much faster alternative with better quality.
    ffmpeg -v quiet -stats -i "$1" -c:v libx265 -crf 19 -preset fast -r 120 "$outfile"

    # using the nvidia enconder, faster but requires a GPU
    #ffmpeg -i "$1" -c:v hevc_nvenc -crf 19 -r 120 "$outfile"

    # copies over the trasncoded file to the name of the original
    mv $outfile $1

  else
    echo "is not mjpeg, skipping"
  fi
}

export -f compress


# iterates the compression funciton for all videose in folder
videofolder="/home/mateo/motion/test_compress_deleteme2/"
find $videofolder -type f -exec bash -c  'compress "$@"' bash {} \;
