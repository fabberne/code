import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.patches as mpatches

from tabulate import tabulate
from matplotlib.transforms import Affine2D

def print_displacements(Unodal):
    print('Nodal displacements and rotations')
    
    u_table_data = [("Node", "Ux [mm]", "Uz [mm]", "Ry [mrad]")]
    
    for node, (ux, uy, Rtheta) in enumerate(Unodal):
        u_table_data.append((node, ux*1000, uy*1000, -Rtheta*1000))
    
    print(tabulate(u_table_data, headers="firstrow", tablefmt="fancy_grid", floatfmt=".2f"))



def print_section_forces(SectionForces):
    print('Section forces')
    
    F_table_data = [("element", "Ns [kN]", "Vs [kN]", "Ms [kNm]", "Ne [kN]", "Ve [kN]", "Me [kNm]")]
    
    for element, (Ns, Vs, Ms, Ne, Ve, Me) in enumerate(SectionForces):
        F_table_data.append((element, Ns, Vs, Ms, Ne, Ve, Me))
    
    print(tabulate(F_table_data, headers="firstrow", tablefmt="fancy_grid", floatfmt=".2f"))



def return_coefficients(ys, ye, L, rs, re, E, I, q):
    C5 = q/(24*E*I)
    C4 = (12*E*I*L*re + 12*E*I*L*rs - 24*E*I*ye + 24*E*I*ys - L**4*q)/(12*E*I*L**3)
    C3 = (-24*E*I*L*re - 48*E*I*L*rs + 72*E*I*ye - 72*E*I*ys + L**4*q)/(24*E*I*L**2)
    C2 = rs
    C1 = ys
    return C5, C4, C3, C2, C1



def calculate_rotation_angle(x_coords, y_coords):
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    return np.degrees(np.arctan2(dy, dx))  # No need to adjust by -90 degrees





def plot_input_system(nodes, connectivity, crossSections, crossSectionProperties):
    ## configuration of the plots
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_title("System")
    
    # Iterate over elements
    for e, (i, j) in enumerate(connectivity):  
        if crossSections[e] != 2 and crossSections[e] != 3:
            x_coords = nodes[(i, j), 0]  # Nodal x-coordinates for element 'e'
            y_coords = nodes[(i, j), 1]  # Nodal y-coordinates for element 'e'
            
            section_index = crossSections[e][0]
            properties = crossSectionProperties[section_index]
        
            # Plot the elements of the undeformed system
            ax.plot(x_coords, y_coords, 
                    color="green", 
                    linewidth=1, 
                    zorder=-1)
            
            # Calculate rotation angle
            rotation_angle = calculate_rotation_angle(x_coords, y_coords)
    
            # Adjust text position above the line
            text_x = np.mean(x_coords) - 0.20 * np.sin(np.radians(rotation_angle))
            text_y = np.mean(y_coords) + 0.20 * np.cos(np.radians(rotation_angle))
    
            text_x2 = np.mean(x_coords) + 0.30 * np.sin(np.radians(rotation_angle))
            text_y2 = np.mean(y_coords) - 0.30 * np.cos(np.radians(rotation_angle))
            
            # Add text annotation
            ax.annotate(f'Element {e}',color="green", 
                        xy=(text_x, text_y), 
                        ha='center', va='center', 
                        rotation=rotation_angle)
    
            ax.annotate(f'A={properties[0]} \n I={properties[1]}', 
                        color="green", 
                        xy=(text_x2, text_y2), 
                        ha='center', va='center', 
                        fontsize=8, 
                        rotation=rotation_angle)
    
    # Plot nodes and numbers
    for i, node in enumerate(nodes):
        ax.scatter(node[0], node [1], color='black')
        
        # Count occurrences of current node
        occurrences = np.where((nodes == node).all(axis=1))[0]
        
        if occurrences[0] == i:  # First occurrence
            ax.annotate(f"{i}", (node[0], node[1]), 
                        xytext=(-10, 10), 
                        textcoords='offset points', ha='right')
        elif occurrences[1] == i:  # Second occurrence
            ax.annotate(f"{i}", (node[0], node[1]), 
                        xytext=( 10, 10), 
                        textcoords='offset points', ha='left' )
        elif occurrences[2] == i:  # Third occurrence
            ax.annotate(f"{i}", (node[0], node[1]), 
                        xytext=(-10,-10), 
                        textcoords='offset points', ha='right')
        elif occurrences[3] == i:  # Fourth occurrence
            ax.annotate(f"{i}", (node[0], node[1]),
                        xytext=( 10,-10), 
                        textcoords='offset points', ha='left' )
    
    range_x = plt.xlim()
    range_y = plt.ylim()

    # Set the limits of the plot
    if range_y[1]-range_y[0] >= (range_x[1]-range_x[0]) * 6 / 10:
        diff = (range_y[1]-range_y[0])*10/6 - (range_x[1]-range_x[0])
        plt.xlim(range_x[0] - diff / 2 - 0.5, range_x[1] + diff /2 + 0.5)
        plt.ylim(range_y[0]            - 0.5, range_y[1]           + 0.5)
    else:
        diff =  (range_x[1]-range_x[0])*6/10 -(range_y[1]-range_y[0])
        plt.xlim(range_x[0]            - 0.5, range_x[1]           + 0.5)
        plt.ylim(range_y[0] - diff / 2 - 0.5, range_y[1] + diff /2 + 0.5)

    # plt.savefig("01_input_system.png", dpi=300)
    plt.show()






def plot_static_system(nodes, connectivity, crossSections, BCs, nodalLoads, distributedLoads):
    ## configuration of the plots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.set_title("Static System")
    
    ## Loop over the rows of the connectivity matrix to plot the elements
    for e, (i, j) in enumerate(connectivity):
        if crossSections[e] not in [2, 3]:
            # Get the nodal displacements of each element
            x_coords = nodes[(i, j), 0]
            y_coords = nodes[(i, j), 1]
        
            ## Plot the elements of the undeformed system
            if crossSections[e] == 1:
                ax.plot(x_coords, y_coords, color="blue", linewidth=2)
            else:
                ax.plot(x_coords, y_coords, color="black", linewidth=2)
    
    # save the current dimensions for later
    range_x = plt.xlim()
    range_y = plt.ylim()
    
    # plot boundary conditions
    for bc, (node, x_bc, y_bc, r_bc, angle) in enumerate(BCs):
        if not (x_bc == 0 and y_bc == 0 and r_bc == 1):
            x_position = nodes[int(node)][0]
            y_position = nodes[int(node)][1]
            x_pos_roated =  x_position * np.cos(angle/180*math.pi) + y_position * np.sin(angle/180*math.pi)
            y_pos_roated = -x_position * np.sin(angle/180*math.pi) + y_position * np.cos(angle/180*math.pi)
        
            if node == 7:
                support_img = image.imread('spring.png')
                scale = 0.0012
            elif x_bc == 1 and y_bc == 1 and r_bc == 1:
                support_img = image.imread('boundary_1_1_1.png')
                scale = 0.001
            elif x_bc == 1 and y_bc == 1 and r_bc == 0:
                support_img = image.imread('boundary_1_1_0.png')
                scale = 0.0015
            elif x_bc == 0 and y_bc == 1 and r_bc == 1:
                support_img = image.imread('boundary_0_1_1.png')
                scale = 0.0015
            elif x_bc == 1 and y_bc == 0 and r_bc == 1:
                support_img = image.imread('boundary_1_0_1.png')
                scale = 0.0015
            elif x_bc == 0 and y_bc == 1 and r_bc == 0:
                support_img = image.imread('boundary_0_1_0.png')
                scale = 0.0015
        
            # Calculate the new extent for the image based on the desired position
            image_width = scale * support_img.shape[1]
            image_height = scale * support_img.shape[0]
            if node == 7:
                extent = (x_pos_roated - image_width / 2, x_pos_roated + image_width / 2, y_pos_roated - image_height/3*2, y_pos_roated + image_height/3)
            else:
                extent = (x_pos_roated - image_width / 2, x_pos_roated + image_width / 2, y_pos_roated - image_height, y_pos_roated)
        
            # Create a transformation to rotate the image
            transform = Affine2D().rotate_deg(angle)
        
            # Plot the rotated image with the new extent
            ax.imshow(support_img, aspect='auto', extent=extent, zorder=-1, transform=transform + ax.transData)
    
    
    # Plot Nodal Loads
    if nodalLoads.size > 0:
        scale_Fx    = max(np.abs((nodalLoads[:,1])))
        scale_Fz    = max(np.abs((nodalLoads[:,2])))
        scale_nodal = max(scale_Fx,scale_Fz) / 3
        
        for bc, (node, F_x, F_z, M_y) in enumerate(nodalLoads):
            x_length = F_x / scale_nodal 
            y_length = F_z / scale_nodal
            
            x_end = nodes[int(node)][0]
            y_end = nodes[int(node)][1]
        
            F     = math.sqrt(F_x**2 + F_z**2)
            if F_x != 0:
                angle = -math.degrees(math.atan(-F_z / F_x))
            else:
                angle = 90
    
            arrow = mpatches.FancyArrowPatch((x_end - x_length, y_end - y_length), 
                                                    (x_end, y_end), mutation_scale=15, color="red", zorder=3)
            ax.add_patch(arrow)
            ax.text(x_end - x_length/2 + 0.4,  
                    y_end - y_length/2 - 0.5, 
                    r"${:.2f}$ $kN$".format(F / 1000) ,
                    fontsize=11, ha='center',color="red",
                    rotation=angle,rotation_mode='anchor')
    
    
    # plot distributed loads
    scale_distri = max(np.abs(np.concatenate((distributedLoads[:,0], distributedLoads[:,1])))) * 1.5
    for element, (q_s, q_e) in enumerate(distributedLoads):
        if np.any(distributedLoads[element] != 0):
            # Get the nodes
            x_coords = nodes[connectivity[element], 0]
            y_coords = nodes[connectivity[element], 1]
        
            # Evaluate and plot section forces (s:start node, e:end node)
            xs, ys = x_coords[0], y_coords[0]
            xe, ye = x_coords[1], y_coords[1]   
            
            # Plotting params to make sure we plot perpendicular to element
            L         = np.sqrt((xe-xs)**2+(ye-ys)**2)
            cos_theta = (xe-xs)/L
            sin_theta = (ye-ys)/L
            angle     = math.degrees(math.asin(sin_theta))
            
            q_xs = xs+q_s*  sin_theta /scale_distri
            q_xe = xe+q_e*  sin_theta /scale_distri
            q_ys = ys+q_s*(-cos_theta)/scale_distri
            q_ye = ye+q_e*(-cos_theta)/scale_distri
            ax.plot((xs, q_xs, q_xe, xe), (ys, q_ys, q_ye,  ye), color="red", linewidth=1)
    
            number_of_arrows = int(L / 0.5)
            arrow_x_local = np.linspace(0   , L   , number_of_arrows)
            arrow_y_local = np.linspace(0   , 0   , number_of_arrows)
            q_linear      = np.linspace(-q_s, -q_e, number_of_arrows) / scale_distri
            
            arrow_x_end   = arrow_x_local * cos_theta - arrow_y_local * sin_theta
            arrow_y_end   = arrow_x_local * sin_theta + arrow_y_local * cos_theta
    
            arrow_x_start = arrow_x_end - q_linear * sin_theta
            arrow_y_start = arrow_y_end + q_linear * cos_theta
    
            offset = [(q_s + q_e)/2*  sin_theta /scale_distri, 
                      (q_s + q_e)/2*(-cos_theta)/scale_distri]
            
            if (q_s + q_e) <= 0:
                offset[0] = offset[0] - 0.2*sin_theta
                offset[1] = offset[1] + 0.2*cos_theta
            else:
                offset[0] = offset[0] + 0.5*sin_theta
                offset[1] = offset[1] - 0.5*cos_theta
    
            for i in range(0, number_of_arrows):
                arrow = mpatches.FancyArrowPatch((arrow_x_start[i] + xs, arrow_y_start[i] + ys), 
                                                (arrow_x_end[i] + xs, arrow_y_end[i] + ys), mutation_scale=5, color="red")
                ax.add_patch(arrow)
                ax.text((xs+xe)/2+offset[0],  
                        (ys+ye)/2+offset[1], 
                        r"${:.2f}$ $kN/m$".format(np.abs((q_s + q_e)/2) / 1000) ,
                        fontsize=11, ha='center',color="red",
                        rotation=angle,rotation_mode='anchor')
    
    # Add custom legend
    custom_handles = [plt.Line2D([], [], color='blue' , linestyle='-'),
                      plt.Line2D([], [], color='black', linestyle='-')]
    custom_labels  = ['beam', 'column']
    plt.legend(custom_handles, custom_labels, loc="lower right")     

    if range_y[1]-range_y[0] >= (range_x[1]-range_x[0]) * 6 / 10:
        diff = (range_y[1]-range_y[0])*10/6 - (range_x[1]-range_x[0])
        plt.xlim(range_x[0] - diff / 2 - 0.5,  range_x[1] + diff /2 + 0.5)
        plt.ylim(range_y[0]- 0.5, range_y[1] + 0.5)
    else:
        diff =  (range_x[1]-range_x[0])*6/10 -(range_y[1]-range_y[0])
        plt.xlim(range_x[0]- 0.5, range_x[1] + 0.5)
        plt.ylim(range_y[0] - diff / 2- 0.5, range_y[1] + diff /2 + 0.5)

    # plt.savefig("02_static_system.png", dpi=300)
    plt.show()






def plot_results(Unodal, SectionForces, connectivity, nodes, crossSections, distributedLoads, crossSectionProperties, E):
        
    ## configuration of the plots
    fig = plt.figure(figsize=(14, 9))
    
    # Create subplots and name them
    displacements   = fig.add_subplot(221)
    normal_forces   = fig.add_subplot(222)
    shear_forces    = fig.add_subplot(223)
    bending_moments = fig.add_subplot(224)
    
    displacements.axis(  'equal')
    normal_forces.axis(  'equal')
    shear_forces.axis(   'equal')
    bending_moments.axis('equal')
    
    displacements.tick_params(  left   = False, labelleft   = False,  
                                bottom = False, labelbottom = False)
    normal_forces.tick_params(  left   = False, labelleft   = False,  
                                bottom = False, labelbottom = False)
    shear_forces.tick_params(   left   = False, labelleft   = False,  
                                bottom = False, labelbottom = False)
    bending_moments.tick_params(left   = False, labelleft   = False,  
                                bottom = False, labelbottom = False)
    
    ## scaling the plots
    # get the highest internal forces and nodal displacements to scale the plots
    U_max = max( 1, max(np.abs(np.concatenate((Unodal[:,0]       , Unodal[:,1]       )))) * 1000)
    N_max = max( 1, max(np.abs(np.concatenate((SectionForces[:,0], SectionForces[:,3])))))
    V_max = max( 1, max(np.abs(np.concatenate((SectionForces[:,1], SectionForces[:,4])))))
    M_max = max( 1, max(np.abs(np.concatenate((SectionForces[:,2], SectionForces[:,5])))))
            
    scale_displ  = int(1000 / U_max)
    scale_normal = int(1.00 * N_max)
    scale_shear  = int(0.44 * V_max)
    scale_moment = int(0.70 * M_max)

    displacements.set_title(r"Displacements [$mm$] (scaling = ${:.0f}$)".format(scale_displ))
    normal_forces.set_title(  "Normal Forces [$kN$]"   )
    shear_forces.set_title(   "Shear Forces [$kN$]"    )
    bending_moments.set_title("Bending Moments [$kNm$]")
    
    ## Loop over the rows of the connectivity matrix
    for e, (i, j) in enumerate(connectivity):
        # Get the nodal displacements of each element
        x_coords = nodes[(i, j), 0]
        y_coords = nodes[(i, j), 1]
    
        # Evaluate and plot section forces (s:start node, e:end node)
        xs, ys = x_coords[0], y_coords[0]
        xe, ye = x_coords[1], y_coords[1]   
        
        # Plotting params to make sure we plot perpendicular to element
        L         = np.sqrt((xe-xs)**2+(ye-ys)**2)
        cos_theta = (xe-xs)/L
        sin_theta = (ye-ys)/L
        angle     = math.degrees(math.asin(sin_theta))
    
        ## Plot the elements of the undeformed system
        displacements.plot(  x_coords, y_coords, color="black", linewidth=0.5)
        normal_forces.plot(  x_coords, y_coords, color="black", linewidth=0.5)
        shear_forces.plot(   x_coords, y_coords, color="black", linewidth=0.5)
        bending_moments.plot(x_coords, y_coords, color="black", linewidth=0.5)
    
        ## Plot the deformed system
        # Get the nodal displacements of each element
        x_displ = Unodal[(i, j), 0]
        y_displ = Unodal[(i, j), 1]
    
        # Update the nodal coordinates of each element
        x_displ_coords = x_coords+scale_displ*x_displ
        y_displ_coords = y_coords+scale_displ*y_displ
    
        # Plot the real displacements
        xs_def_global = xs+x_displ[0]
        xe_def_global = xe+x_displ[1]
        ys_def_global = ys+y_displ[0]
        ye_def_global = ye+y_displ[1]
        rs, re = Unodal[i][2], Unodal[j][2]

        xs_local =  xs*cos_theta + ys*sin_theta
        xe_local =  xe*cos_theta + ye*sin_theta
        ys_local = -xs*sin_theta + ys*cos_theta
        ye_local = -xe*sin_theta + ye*cos_theta
        xs_def_local = xs_local + x_displ[0]*cos_theta + y_displ[0]*sin_theta
        xe_def_local = xe_local + x_displ[1]*cos_theta + y_displ[1]*sin_theta
        ys_def_local = ys_local - x_displ[0]*sin_theta + y_displ[0]*cos_theta
        ye_def_local = ye_local - x_displ[1]*sin_theta + y_displ[1]*cos_theta
        
        I          =  crossSectionProperties[crossSections[e]][0][1]
        l_deformed = (xe_def_local-xe_local)*scale_displ + xe_local - ((xs_def_local-xs_local)*scale_displ + xs_local)

        q_s   = distributedLoads[e][0]
        q_e   = distributedLoads[e][1]
        q_avg = (q_s+q_e)/2

        C5, C4, C3, C2, C1 = return_coefficients(0, ye_def_local-ys_def_local, l_deformed, rs, re, E, I, q_avg)
            
        x_local = np.linspace(0, l_deformed, 50)
        y_local = (C5*x_local**4 + C4*x_local**3 + C3*x_local**2 + C2*x_local + C1)*scale_displ

        x_global = x_local * cos_theta - y_local * sin_theta
        y_global = x_local * sin_theta + y_local * cos_theta
        
        displacements.plot(xs + (xs_def_global-xs)*scale_displ + x_global,
                    ys + (ys_def_global-ys)*scale_displ + y_global, 
                    label="actual displacements", color="red", linewidth=2.5)
            
        # Plot the elements of the deformed system linear
        displacements.plot(x_displ_coords, y_displ_coords, color="blue", linewidth=1.2)
        custom_handles = [plt.Line2D([], [], color='blue', linestyle='-'),
                            plt.Line2D([], [], color='red', linestyle='-')]
        custom_labels  = ['linear deforamtion', 'actual deforamtion']
        displacements.legend(custom_handles, custom_labels, loc="lower right")   
        
    
        ## Plot the section forces
        # get the section forces
        Ns, Ne =  SectionForces[e,0],  SectionForces[e,3]
        Vs, Ve = -SectionForces[e,1], -SectionForces[e,4]
        Ms, Me =  SectionForces[e,2],  SectionForces[e,5]
    
        Nxs,Nxe = xs-Ns*  sin_theta /scale_normal, xe-Ne*  sin_theta /scale_normal
        Nys,Nye = ys-Ns*(-cos_theta)/scale_normal, ye-Ne*(-cos_theta)/scale_normal
        Vxs,Vxe = xs-Vs*  sin_theta /scale_shear , xe-Ve*  sin_theta /scale_shear
        Vys,Vye = ys-Vs*(-cos_theta)/scale_shear , ye-Ve*(-cos_theta)/scale_shear
        Mxs,Mxe = xs-Ms*  sin_theta /scale_moment, xe-Me*  sin_theta /scale_moment
        Mys,Mye = ys-Ms*(-cos_theta)/scale_moment, ye-Me*(-cos_theta)/scale_moment
    
        # PLot the normal Forces
        fontsize=10
        offset = [-Ns*  sin_theta /scale_normal, 
                    -Ns*(-cos_theta)/scale_normal]
        if Ns >= 0:
            # for elements in Tension
            color="gold"
            offset[0] = offset[0] - 0.2*sin_theta
            offset[1] = offset[1] + 0.2*cos_theta
        else:
            # for elements under pressure
            color="blue"
            offset[0] = offset[0] + 0.5*sin_theta
            offset[1] = offset[1] - 0.5*cos_theta
    
        normal_forces.plot((xs, Nxs, Nxe, xe), (ys, Nys, Nye,  ye), color=color, linewidth=1)
        normal_forces.fill((xs, Nxs, Nxe, xe), (ys, Nys, Nye,  ye), color=color , alpha=0.2)
        normal_forces.text((xs+xe)/2+offset[0], (ys+ye)/2+offset[1], r"${:.2f}$".format(Ns), fontsize=fontsize, ha='center',color=color, rotation=angle,rotation_mode='anchor')

        custom_handles = [plt.Line2D([], [], color='gold', linestyle='-'),
                            plt.Line2D([], [], color='blue', linestyle='-')]
        custom_labels  = ['tension', 'compression']
        normal_forces.legend(custom_handles, custom_labels, loc="lower right")      
    
        # PLot the shear Forces
        V_max = max(abs(Vs), abs(Ve))
        color = "purple"
        if Vs != 0 or Ve != 0:
            # plot the shear forces
            shear_forces.plot((xs, Vxs, Vxe, xe), (ys, Vys, Vye,  ye), color=color, linewidth=1)
            shear_forces.fill((xs, Vxs, Vxe, xe), (ys, Vys, Vye,  ye), color=color , alpha=0.2)
    
            # write the value of the highest shear force near to it
            if V_max == abs(Vs):
                offset = [-Vs*  sin_theta /scale_shear + 0.2*cos_theta, 
                            -Vs*(-cos_theta)/scale_shear + 0.2*sin_theta]
                if Vs >= 0:
                    offset[0] -= 0.2*sin_theta
                    offset[1] += 0.2*cos_theta
                else:
                    offset[0] += 0.5*sin_theta
                    offset[1] -= 0.5*cos_theta
                shear_forces.text((xs)+offset[0], (ys)+offset[1], r"${:.2f}$".format(Vs), fontsize=fontsize, ha='left',color=color, rotation=angle,rotation_mode='anchor')
            else:
                offset = [-Ve*  sin_theta /scale_shear - 0.2*cos_theta, 
                            -Ve*(-cos_theta)/scale_shear - 0.2*sin_theta]
                if Ve >= 0:
                    offset[0] -= 0.2*sin_theta
                    offset[1] += 0.2*cos_theta
                else:
                    offset[0] += 0.5*sin_theta
                    offset[1] -= 0.5*cos_theta
                shear_forces.text((xe)+offset[0], (ye)+offset[1], r"${:.2f}$".format(Ve), fontsize=fontsize, ha='right',color=color, rotation=angle,rotation_mode='anchor')
        
        # PLot the bending Moments
        M_max = max(abs(Ms), abs(Me))
        color = "red"
        if Ms != 0 or Me != 0:
            # plot the bending moments quadratically if there is a distributed load
            if np.any(distributedLoads[e] != 0):
                q_s   = distributedLoads[e][0]
                q_e   = distributedLoads[e][1]
                q_avg = (q_s+q_e)/2
                
                num = 50
                l_values      = np.linspace(0, L, num)
                moment_values = -np.array([Ms + Vs * x + (-q_avg/1000)*x**2 / 2 for x in l_values])
                #moment_values = -moment_values[::-1]
    
                x_values = np.linspace(xs, xe, num)
                y_values = np.linspace(ys, ye, num)
    
                Mx = [xs]
                My = [ys]
                for k in range(0,num):
                    Mx.append(x_values[k] + moment_values[k] * sin_theta /scale_moment )
                    My.append(y_values[k] - moment_values[k] * cos_theta /scale_moment )
                Mx.append(xe)
                My.append(ye)
                
                bending_moments.plot(Mx, My, color=color, linewidth=1)
                bending_moments.fill(Mx, My, color=color , alpha=0.2)
            else:
                bending_moments.plot((xs, Mxs, Mxe, xe), (ys, Mys, Mye,  ye), color=color, linewidth=1)
                bending_moments.fill((xs, Mxs, Mxe, xe), (ys, Mys, Mye,  ye), color=color , alpha=0.2)
                
            # write the value of the highest bending moment to it 
            if M_max == abs(Ms):
                offset = [-Ms*  sin_theta /scale_moment + 0.2*cos_theta, 
                            -Ms*(-cos_theta)/scale_moment + 0.2*sin_theta]
                if Ms >= 0:
                    offset[0] -= 0.2*sin_theta
                    offset[1] += 0.2*cos_theta
                else:
                    offset[0] += 0.3*sin_theta
                    offset[1] -= 0.3*cos_theta
                bending_moments.text((xs)+offset[0], (ys)+offset[1], r"${:.2f}$".format(Ms), fontsize=fontsize, ha='left',color=color, rotation=angle,rotation_mode='anchor')
            else:
                offset = [-Me*  sin_theta /scale_moment - 0.2*cos_theta, 
                            -Me*(-cos_theta)/scale_moment - 0.2*sin_theta]
                if Me >= 0:
                    offset[0] -= 0.2*sin_theta
                    offset[1] += 0.2*cos_theta
                else:
                    offset[0] += 0.3*sin_theta
                    offset[1] -= 0.3*cos_theta
                bending_moments.text((xe)+offset[0], (ye)+offset[1], r"${:.2f}$".format(Me), fontsize=fontsize, ha='right',color=color, rotation=angle,rotation_mode='anchor')

    ## write the maximum deformation
    # get the index of the greatest deformation
    if np.all(Unodal[:, 0] == 0) and np.all(Unodal[:, 1] == 0):
        node = [0]
    else:
        node, x_y = np.where(np.abs(Unodal)*1000 == U_max)
    # get the coordinates and displacements
    x_coord, y_coord = nodes[node[0]]
    x_displ, z_displ = Unodal[node, 0][0]*1000, Unodal[node, 1][0]*1000
    # plot the text
    displacements.text(x_coord+x_displ/1000*scale_displ-0.2, y_coord+z_displ/1000*scale_displ -0.4, r"Ux: ${:.2f}$".format(x_displ), fontsize=fontsize, ha='center',color="blue")
    displacements.text(x_coord+x_displ/1000*scale_displ-0.2, y_coord+z_displ/1000*scale_displ -0.9, r"Uz: ${:.2f}$".format(z_displ), fontsize=fontsize, ha='center',color="blue")


    plt.tight_layout()
    # plt.savefig("03_solution.png", dpi=300)
    plt.show()
