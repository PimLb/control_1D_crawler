import numpy as np
from env import Environment
from learning import actionValue
from analysis_utilities import *


random_policy = False
constrainTIP = False

RENDER = False
#PARAMETERS
sim_shape = (21,)
t_position=100
omega = 0.1

int_steps = 20000
sim_vel = []

standardHive = int(input("insert 1 for getting standard hive"))
if standardHive:



    if RENDER:
        print("ANIMATION ON 12 suckers")
        n_suckers = 12
        env = Environment(n_suckers,sim_shape,t_position, omega =omega)
        Q = actionValue(env.info,hiveUpdate=True)
        hivePol = {'->|<-': 0, '->|->': 0, '<-|<-': 1, '<-|->': 0, 'base|<-': 1, 'base|->': 0, '->|tip': 0, '<-|tip': 1}
        Q.loadPolicy(hivePol) #<-- used to constrain base and tip to correct policy
        state = env.get_state()
        for k in range(10000):
            onPolAction = Q.getOnPolicyAction(state)
            state,_r,_t = env.step(onPolAction)
            if k %10 ==0:
                env.render() 
    #INFO:
    # Periodic refers to being a periodic tentacle --> different simulator

    #N suckers to check
    NS = [5, 8, 10, 12, 15, 20, 25, 30, 35]
    for n_suckers in NS:
        env = Environment(n_suckers,sim_shape,t_position, omega =omega) #implies no GANGLIA and OVERDAMPED
       
        Q = actionValue(env.info,hiveUpdate=True)
        hivePol = {'->|<-': 0, '->|->': 0, '<-|<-': 1, '<-|->': 0, 'base|<-': 1, 'base|->': 0, '->|tip': 0, '<-|tip': 1}
        Q.loadPolicy(hivePol) #<-- used to constrain base and tip to correct policy
        
        env.equilibrate(1000)
        state = env.get_state()
        for k in trange(int_steps):
            onPolAction = Q.getOnPolicyAction(state)

            state,r,t=env.step(onPolAction)

            

        print("average velocity (normalized) =",env.get_averageVel()/env.x0)
        sim_vel.append(env.get_averageVel()/env.x0)


    print("RESULT VELS:")
    print(NS)
    print(sim_vel)
        

else:

    if RENDER:
        print("ANIMATION ON 12 suckers")
        n_suckers = 12
        env = Environment(n_suckers,sim_shape,t_position, omega =omega)
        Q = actionValue(env.info,hiveUpdate=True)
        hivePol = {'->|<-': 0, '->|->': 0, '<-|<-': 1, '<-|->': 0, 'base|<-': 1, 'base|->': 0, '->|tip': 0, '<-|tip': 1}
        Q.loadPolicy(hivePol) #<-- used to constrain base and tip to correct policy
        state = env.get_state()
        for k in range(10000):
            onPolAction = Q.getOnPolicyAction(state)
            action = [0]*n_suckers
            if not random_policy:
                ns,ids = optimum_impulse(env._t,env.omega,env.N,env._nsuckers)
                #print(s_id) 
                for s in ids:
                    action[s]=1
            else:
                for n in range(n_suckers):
                    action[n] = np.random.randint(0,2)
            if  constrainTIP:
                action[0] = onPolAction[0]#base
                action[-1] = onPolAction[-1]#tip
            state,_r,_t = env.step(action)
            if k %10 ==0:
                env.render() 
    #INFO:
    # Periodic refers to being a periodic tentacle --> different simulator

    #N suckers to check
    NS = [5, 8, 10, 12, 15, 20, 25, 30, 35]
    if random_policy and constrainTIP:
        print ("Extracting reference RANDOM POLICY WITH CLEVER TIP BASE")
    elif random_policy and not constrainTIP:
        print ("Extracting reference RANDOM POLICY (complete)")
    else:
        print ("Extracting reference semi-analytical best single inpulse policy")
    for n_suckers in NS:
        env = Environment(n_suckers,sim_shape,t_position, omega =omega) #implies no GANGLIA and OVERDAMPED
        if env.info["isPeriodic"]:
            periodic = True
            type_of_T = "periodic tentacle"
        else:
            periodic =False
            type_of_T = "finite tentacle"
            Q = actionValue(env.info,hiveUpdate=True)
            hivePol = {'->|<-': 0, '->|->': 0, '<-|<-': 1, '<-|->': 0, 'base|<-': 1, 'base|->': 0, '->|tip': 0, '<-|tip': 1}
            Q.loadPolicy(hivePol) #<-- used to constrain base and tip to correct policy
        
        env.equilibrate(1000)
        state = env.get_state()
        for k in trange(int_steps):
            action = [0]*n_suckers
            if periodic ==False:
                onPolAction = Q.getOnPolicyAction(state)
            # print(state)
            # print(onPolAction)
            
            if not random_policy:
                ns,ids = optimum_impulse(env._t,env.omega,env.N,env._nsuckers)
                #print(s_id) 
                for s in ids:
                    action[s]=1
            else:
                for n in range(n_suckers):
                    action[n] = np.random.randint(0,2)
                # print(action)
                # input()
            
            if periodic == False and constrainTIP:
                action[0] = onPolAction[0]#base
                action[-1] = onPolAction[-1]#tip
            # print(action)
            
            state,r,t=env.step(action)

            

        print("average velocity (normalized) =",env.get_averageVel()/env.x0)
        sim_vel.append(env.get_averageVel()/env.x0)


    print("RESULT VELS:")
    print(NS)
    print(sim_vel)
        
