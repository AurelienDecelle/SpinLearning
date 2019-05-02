import numpy as np

getindex={}



def compute_connectivity(Lx,Ly):

	pos=np.zeros((Lx,Ly))
	index=0
	for y in range(Ly):
		for x in range(Lx):
			pos[x,y]=int(index)
			getindex[index]=(x,y)
			index+=1
			

	connectivity={}
	for y in range(Ly):
		for x in range(Lx):
			connectivity[(min(pos[x,y],pos[(x+1)%Lx,y]),max(pos[x,y],pos[(x+1)%Lx,y]))] = 1
			connectivity[(min(pos[x,y],pos[x,(y+1)%Ly]),max(pos[x,y],pos[x,(y+1)%Ly]))] = 1


	return connectivity,getindex

def getXY(ind):
    duple=getindex[ind]
    return (duple[0],duple[1]) 

def getPos_Chess_2D(x_1,y_1,Lx,Ly):
    (x1,y1)=x_1
    (x2,y2)=y_1
    n_x = (x1+x2)
    n_y = (y1+y2)
    if(abs(x1-x2)>1):
        n_x = max(x1,x2)+Lx
    if(abs(y1-y2)>1):
        n_y = max(y1,y2)+Ly
    return (int(n_x),int(n_y))


def SplitSet(X,y,r):
    NS = X.shape[0]
    TotTrain = int(NS*r)

    X_train = X[:TotTrain,:]
    y_train = y[:TotTrain]
    # y2_train = y2[:TotTrain]

    X_test = X[TotTrain:,:]
    y_test = y[TotTrain:]
    # y2_test = y2[TotTrain:]
    
    
    # return X_train,y_train,y2_train,X_test,y_test,y2_test
    return X_train,y_train,X_test,y_test

def createSample_2D(datJ,Lx,Ly):
   
    Chess=np.zeros((2*Lx,2*Ly))
    for i in datJ:
        rdm = np.random.randint(0,2)*2-1
        n=getPos_Chess_2D(getXY(i[0]),getXY(i[1]),Lx,Ly)
        Chess[n[0],n[1]]=rdm
  
    return Chess

def getOrbit_2D(Chess,datJ,Lx,Ly):
    Orbit=np.zeros((2*Lx,2*Ly))
    epsilon = np.random.randint(2,size=Lx*Ly)*2-1
    for i in datJ:
        n=getPos_Chess_2D(getXY(i[0]),getXY(i[1]),Lx,Ly)
        Orbit[n[0],n[1]]=Chess[n[0],n[1]]*epsilon[int(i[0])]*epsilon[int(i[1])]  
    return Orbit

def getRandom_2D(Chess,datJ,q,Lx,Ly):
    
    Random=np.zeros((2*Lx,2*Ly))
    nchanges=int(q*len(datJ))
    #print(nchanges)
    if nchanges==0:
        print("Atencion: nada cambia")
    epsilon=np.ones(len(datJ)-nchanges)
    epsilon=np.append(epsilon,-np.ones(nchanges))
    epsilon=np.random.permutation(epsilon)

    cont=0
    for i in datJ:
        n=getPos_Chess_2D(getXY(i[0]),getXY(i[1]),Lx,Ly)
        
        Random[n[0],n[1]]=Chess[n[0],n[1]]*epsilon[cont]
        cont+=1
        
    return Random



def getLine_2D(Chess,datJ,Lx,Ly):
    Line=np.zeros((2*Lx,2*Ly))
    Line[:,:]=Chess[:,:]
    
    if(np.random.random()<0.5):
        k=np.random.randint(0,Lx)
        il=(2*(k)+1)%(int(2*Lx))
        Line[il,:]*=-1
    else:
        k=np.random.randint(0,Ly)
        ic=(2*(k)+1)%(int(2*Ly))
        Line[:,ic]*=-1
    
    return Line


def getLine_loop(Chess,n_x,n_y,Lx,Ly):
    Line=np.zeros((2*Lx,2*Ly))
    Line[:,:]=Chess[:,:]
    ip=int(np.random.random()*Lx*Ly)
    x=[]
    x_0=getXY(ip)
    i=0
    for j in range(n_y):
        x.append(((x_0[0]+i)%Lx,(x_0[1]+j)%Ly))
    for i in range(1,n_x):
        x.append(((x_0[0]+i)%Lx,(x_0[1]+j)%Ly))
    for j in range(n_y):
        x.append(((x_0[0]+i)%Lx,(x_0[1]+n_y-1-j)%Ly))
    for i in range(n_x):
        x.append(((x_0[0]+n_x-1-i)%Lx,(x_0[1]+n_y-1-j)%Ly))
    
    pos=[]
    for i in range(len(x)):
        pos.append(getPos_Chess_2D(x[i],x[(i+1)%len(x)],Lx,Ly))
    
    
    for ip in pos:
        Line[ip[0],ip[1]]*=-1
        
    return Line


def listPlaquettes(Chess,datJ,Lx,Ly):
    m2Lx=int(2*Lx)
    m2Ly=int(2*Ly)
    listPlaq=np.zeros((Lx,Ly))
    sumfrus=0
    
    for x in range(0,m2Lx,2):
        for y in range(0,m2Ly,2):                   # *  J3  *
            J1=Chess[(x+1)%m2Lx,y]                  # J2  * J4
            J2=Chess[x,(y+1)%(m2Ly)]                # *  J1  * 
            J3=Chess[(x+1)%(m2Lx),(y+2)%m2Ly]
            J4=Chess[(x+2)%m2Lx,(y+1)%(m2Ly)]
            prod=J1*J2*J3*J4
            listPlaq[(int(x/2.),int(y/2.))]=prod
            if(prod<0):
                sumfrus+=prod    
  
    return listPlaq,sumfrus

def listLines(Chess,datJ,Lx,Ly):
    m2Lx=int(2*Lx)
    m2Ly=int(2*Ly)
    list_cols=np.zeros((1,Lx))
    list_rows=np.zeros((Ly,1))
    
    for x in range(0,m2Lx,2):
        prod=1
        for y in range(0,m2Ly,2):
            J=Chess[x,(y+1)%(m2Ly)]
            prod*=J
        list_rows[(int(x/2.),0)]=prod
    for y in range(0,m2Ly,2):
        prod=1
        for x in range(0,m2Lx,2):
            J=Chess[(x+1)%(m2Lx),y]
            prod*=J
        list_cols[(0,int(y/2.))]=prod
    
  
    return list_cols,list_rows


def write_PBC(Chess,Lx,Ly):
    m2Lx=int(2*Lx)
    m2Ly=int(2*Ly)
    Chessnew=np.zeros((m2Lx+1,m2Ly+1))
    
    inter=np.tile(Chess[:,:],[3,3])
    Chessnew[:,:]=inter[m2Lx-1:m2Lx*2,m2Ly-1:m2Ly*2]

    return Chessnew

def gauge_fixing_Coulomb(Chess,datJ,Lx,Ly):
    Orbit=np.zeros((2*Lx,2*Ly))
    changed = np.zeros((Lx,Ly)) 
    epsilon = np.ones((Lx,Ly)) 
    
    y=0;
    for x in range(Lx):
        n=getPos_Chess_2D(((x+1)%Lx,y),(x,y),Lx,Ly)
       
        if(changed[(x+1)%Lx,y]==0):
            Jold=Chess[n[0],n[1]]
            epsilon[(x+1)%Lx,y]=Jold*epsilon[x,y]
            changed[(x+1)%Lx,y]=1
    
    for x in range(Lx):
        for y in range(Ly):
            n=getPos_Chess_2D((x,(y+1)%Ly),(x,y),Lx,Ly)
        
            if(changed[x,(y+1)%Ly]==0):
                Jold=Chess[n[0],n[1]]
                epsilon[x,(y+1)%Ly]=Jold*epsilon[x,y]
                changed[x,(y+1)%Ly]=1
    
    for i in datJ:
        n=getPos_Chess_2D(getXY(i[0]),getXY(i[1]),Lx,Ly)

        Orbit[n[0],n[1]]=Chess[n[0],n[1]]*epsilon[getXY(i[0])]*epsilon[getXY(i[1])]
        
    return Orbit

def getLoop(Chess,nl,Lx,Ly):
    Line=np.zeros((2*Lx,2*Ly))
    Line[:,:]=Chess[:,:]

    diff=0
    while (diff==0):
      #  print(diff)
        
        steps=[]
        for i in range(2):
            st = np.ones(nl[i])
            st = np.append(st,-np.ones(nl[i]))
            st = np.random.permutation(st)
            steps.append(st)

        orden = np.zeros(2*nl[0])
        orden = np.append(orden,np.ones(2*nl[1]))
        orden = np.random.permutation(orden)

        ip=int(np.random.random()*Lx*Ly)

        x=[]
        nx=0
        ny=0
        x_0=getXY(ip)
        x.append((x_0[0],x_0[1]))
        for i in range(len(orden)):
            if(orden[i]==0):
                ix=steps[0][nx]
                x_1=((x_0[0]+ix)%Lx,x_0[1]%Ly)
                x.append(x_1)
                nx+=1
            else:
                iy=steps[1][ny]
                x_1=(x_0[0]%Lx,(x_0[1]+iy)%Ly)
                x.append(x_1)
                ny+=1
            x_0=x_1

        pos=[]
        for i in range(len(x)):
            pos.append(getPos_Chess_2D(x[i],x[(i+1)%len(x)],Lx,Ly))


        for ip in pos:
            Line[ip[0],ip[1]]*=-1

        
        for ix in range(2*Lx):
            for iy in range(2*Ly):
                if(Chess[ix,iy]!=Line[ix,iy]):
                    diff+=1

    return Line


def getRandomLine(Chess,nl,Lx,Ly):
    Line=np.zeros((2*Lx,2*Ly))
    Line[:,:]=Chess[:,:]

    diff=0
    while (diff==0):
       # print(diff)
        
        steps=[]
        for i in range(2):
            st = np.random.binomial(1,0.5,size=(nl[i]))
            steps.append(st)

        orden = np.zeros(nl[0])
        orden = np.append(orden,np.ones(nl[1]))
        orden = np.random.permutation(orden)

        ip=int(np.random.random()*Lx*Ly)

        x=[]
        nx=0
        ny=0
        x_0=getXY(ip)
        x.append((x_0[0],x_0[1]))
        for i in range(len(orden)):
            if(orden[i]==0):
                ix=steps[0][nx]
                x_1=((x_0[0]+ix)%Lx,x_0[1]%Ly)
                x.append(x_1)
                nx+=1
            else:
                iy=steps[1][ny]
                x_1=(x_0[0]%Lx,(x_0[1]+iy)%Ly)
                x.append(x_1)
                ny+=1
            x_0=x_1

        pos=[]
        for i in range(len(x)-1):
            pos.append(getPos_Chess_2D(x[i],x[(i+1)%len(x)],Lx,Ly))


        for ip in pos:
            Line[ip[0],ip[1]]*=-1

        
        for ix in range(2*Lx):
            for iy in range(2*Ly):
                if(Chess[ix,iy]!=Line[ix,iy]):
                    diff+=1

    return Line

