echo "stop :$1"
ps -aux|grep $1|grep -v grep|awk -F' ' '{print $2}'|xargs kill -9
