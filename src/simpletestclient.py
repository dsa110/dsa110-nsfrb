import socket
import numpy as np




#create socket object
print("creating socket...",end='')
servSockD = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
print("Done!")

#bind to port number
port = 8843
print("binding to port " + str(port) + "...",end='')
servSockD.bind(('', port))
print("Done!")

#listen for conections
print("listening for connections...",end='')
servSockD.listen(1)
print("Made connection")

totalbytes = 0
while True:
    print("accepting connection...",end='')
    clientSocket,address = servSockD.accept()
    recstatus = 1
    fullMsg = ""
    print("Done!")
    print("Receiving data...")
    while recstatus > 0:
        (strData, ancdata, msg_flags, address) = clientSocket.recvmsg(255)
        recstatus = len(strData)

        #print(strData.hex(),end='')
        fullMsg += strData.hex()
        totalbytes += recstatus
    print("Done! Total bytes read:",totalbytes)

    #convert to numpy array
    print(bytes.fromhex(fullMsg)[:128])
    fullMsgBytes = bytes.fromhex(fullMsg)
    #print(fullMsgBytes[7:256+7].decode('utf-8'))
    print(np.frombuffer(fullMsgBytes[128:]).reshape(32,32,25,16))
    #print(fullMsgBytes[:])
    #print(fullMsgBytes[:].decode('utf-8'))

    clientSocket.close()

    break

print("done",totalbytes)

