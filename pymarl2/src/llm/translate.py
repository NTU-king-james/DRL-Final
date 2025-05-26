import numpy as np
from pysc2.lib.units import get_unit_type
def get_ally_unit_type(env,unit_id):
    if unit_id == env.marine_id:
        return "Marine"
    elif unit_id == env.stalker_id:
        return "Stalker"
    elif unit_id == env.zealot_id:
        return "Zealot"
    elif unit_id == env.colossus_id:
        return "Colossus"
    elif unit_id == env.marauder_id:
        return "Marauder"
    elif unit_id == env.medivac_id:
        return "Medivac"
    elif unit_id == env.hydralisk_id:
        return "Hydralisk"
    elif unit_id == env.baneling_id:
        return "Baneling"
    elif unit_id == env.zergling_id:
        return "Zergling"
    else:
        return "Unknown"
def get_state_NL(env, state):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if env.obs_instead_of_state:
            # why?
            pass
        s = ''
        base_idx = 0
        enemy_count = env.get_obs_enemy_feats_size()[0]
        ally_count = env.get_obs_ally_feats_size()[0] + 1
        nf_al = env.get_ally_num_attributes()
        nf_en = env.get_enemy_num_attributes()
        # Ally features
        for e in range(ally_count):
            feats = state[base_idx:base_idx+nf_al] 
            base_idx += nf_al
            s += f'ALLY #{e+1}:\n'
            if np.all(feats == 0):
                s += 'Dead.\n'
                continue
            e_dict = {'health':feats[0],'weapon cooldown':feats[1],'x':feats[2],'y':feats[3],'unit':env.agents[e].unit_type}
            ind = 4
            if env.shield_bits_ally > 0:
                e_dict['shield'] = feats[ind]
                ind += 1
            if env.unit_type_bits > 0:
                ind += 1
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield' or k == 'weapon cooldown':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_ally_unit_type(env,v)}\n'
                elif k == 'x' or k == 'y':
                    s += f'relative {k} : {v:.3f}\n'
        for e in range(enemy_count):
            feats = state[base_idx:base_idx+nf_en] 
            base_idx += nf_en
            s += f'ENEMY #{e+1}:\n'
            if np.all(feats == 0):
                s += 'Dead.\n'
                continue
            e_dict = {'health':feats[0],'x':feats[1],'y':feats[2],'unit':env.enemies[e].unit_type}
            ind = 4
            if env.shield_bits_enemy > 0:
                e_dict['shield'] = feats[ind]
                ind += 1
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield' or k == 'weapon cooldown':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_unit_type(v).name}\n'
                elif k == 'x' or k == 'y':
                    s += f'relative {k} : {v:.3f}\n'
        
        if env.state_last_action:
            pass
        if env.state_timestep_number:
            pass
        return s
def get_obs_agent_NL(env, obs, agent_id):
        enemy_feats_dim = env.get_obs_enemy_feats_size()
        ally_feats_dim = env.get_obs_ally_feats_size()
        if np.all(obs == 0):
            return None # agent is dead
        base_idx = 4
        dir_list = ["north","south","east","west"]
        s = ''
        s += 'Available move directions:\n'+"\n".join([dir_list[i] for i in range(4) if obs[i] > 0])+'\n'
        # if you want...
        '''ind = env.n_actions_move

        if env.obs_pathing_grid:
            move_feats[
                ind : ind + env.n_obs_pathing  # noqa
            ] = env.get_surrounding_pathing(unit)
            ind += env.n_obs_pathing

        if env.obs_terrain_height:
            move_feats[ind:] = env.get_surrounding_height(unit)'''
        s += 'ENEMY STATISTICS(visible only):\n'
        # Enemy features
        for e in range(enemy_feats_dim[0]):
            
            feats = obs[base_idx:base_idx+enemy_feats_dim[1]] 
            base_idx += enemy_feats_dim[1]
            s += f'ENEMY #{e+1}:\n'
            if np.all(feats == 0):
                s += 'Not visible or dead.\n'
                continue
            e_dict = {'attack':feats[0],'distance':feats[1],'x':feats[2],'y':feats[3],'unit':env.enemies[e].unit_type}
            ind = 4
            if env.obs_all_health:
                e_dict['health'] = feats[ind]
                ind += 1
                if env.shield_bits_enemy > 0:
                    e_dict['shield'] = feats[ind]
                    ind += 1
                if env.unit_type_bits > 0:
                    ind += 1
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_unit_type(v).name}\n'
                elif k == 'x' or k == 'y':
                    s += f'relative {k} : {v:.3f}\n'
                elif k == 'distance':
                    s += f'distance : {v:.3f}\n'
                elif k == 'attack':
                    s += f'Is attackable : {v > 0}\n'
            
            
        s += '\nALLY STATISTICS(visible only):\n'
        for a in range(ally_feats_dim[0]):
            feats = obs[base_idx:base_idx+ally_feats_dim[1]] 
            base_idx += ally_feats_dim[1]
            s += f'ALLY #{a+1}:\n'
            if np.all(feats == 0):
                s += 'Not visible or dead.\n'
                continue
            e_dict = {'visible':feats[0],'distance':feats[1],'x':feats[2],'y':feats[3],'unit':env.agents[a if a < agent_id else a+1].unit_type}
            ind = 4
            if env.obs_all_health:
                e_dict['health'] = feats[ind]
                ind += 1
                if env.shield_bits_ally > 0:
                    e_dict['shield'] = feats[ind]
                    ind += 1
            if env.unit_type_bits > 0:
                ind += 1
            #if env.obs_last_action:
            #    if you want...?
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_ally_unit_type(env,v)}\n'
                elif k == 'x' or k == 'y':
                    s += f'relative {k} : {v:.3f}\n'
                elif k == 'distance':
                    s += f'distance : {v:.3f}\n'
                elif k == 'visible': # it's always visible ......
                    continue 
        
        # Own features
        s += '\nAgent information:\n'
        if env.obs_own_health:
            s += f'Health : {100*obs[base_idx]:.1f}%\n'
            base_idx += 1
        s += f'unit type : {get_ally_unit_type(env,env.agents[agent_id].unit_type)}\n'
        base_idx += 1
        return s
def get_obs_NL(env,obs_list):
    return [get_obs_agent_NL(env,obs,i) for i,obs in enumerate(obs_list)]