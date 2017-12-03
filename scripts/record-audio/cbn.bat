cd C:\Users\nazli\Desktop\ffmpeg-20170830-2b9fd15-win64-static
cd bin

SET HOUR=%time:~0,2%

set datetimef9=cbn_%date:~-4%_%date:~3,2%_%date:~0,2%__%time:~0,2%_%time:~3,2%_%time:~6,2%.wav
set datetimef24=cbn_%date:~-4%_%date:~3,2%_%date:~0,2%__%time:~1,1%_%time:~3,2%_%time:~6,2%.wav

if "%HOUR:~0,1%" == " " (SET dtStamp=%datetimef24%) else (SET dtStamp=%datetimef9%)

ffmpeg -i rtmp://media.sgr.globo.com:1935/CBNRD/cbnsp.sdp -t 20 %dtStamp%
