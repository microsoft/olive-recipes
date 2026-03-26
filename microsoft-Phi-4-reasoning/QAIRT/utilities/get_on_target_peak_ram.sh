PROCESS=$1
echo $PROCESS
if [ "$PROCESS" == "" ]; then
    exit 0
fi
PEAK=0
while true; do 
    PID=$(pgrep $PROCESS)
    if [ "$PID" != "" ]; then
        DMA=$(dmabuf_dump $PID | grep "PROCESS TOTAL" | awk '{ print $3 }')
        PSS=$(dumpsys meminfo -s $PID | grep "TOTAL PSS" | awk '{ print $3 }')
        if [ "$PSS" == "" ] || [ "$PSS" == "" ]; then
            continue
        fi
        TOTAL=$(($DMA+$PSS))
        if [ TOTAL -gt PEAK ]; then
            PEAK=$TOTAL
            echo "New PEAK : $PEAK (DMA:$DMA, PSS:$PSS)"
            log -t PEAKMEM "$PEAK (DMA:$DMA, PSS:$PSS)"
        fi
    fi 
done