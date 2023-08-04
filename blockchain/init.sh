for port in $@
do
    python blockchain.py --port $port &
done
sleep 5
python register.py --ports $@
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
