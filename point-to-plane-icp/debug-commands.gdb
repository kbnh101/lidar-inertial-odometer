# gdb helper commands for inspecting PointToPlaneCostFunction::Evaluate().
#
# Sourced by debug.gdb (terminal use) and by .vscode/launch.json (F5 use).
# Only defines commands -- it sets no breakpoints, so VSCode's gutter breakpoints stay in charge.

set print pretty on

# Eigen keeps its coefficients in m_storage.m_data.array, which prints as a plain list.
define evec
    print $arg0.m_storage.m_data.array
end
document evec
Print an Eigen fixed-size vector as a plain coefficient list.  Usage: evec source_point_
end

# Run this while stopped anywhere inside Evaluate(), after the Jacobian has been filled.
define jac
    printf "\n--- correspondence -------------------------------------------\n"
    printf "x_n (source) = [% .6f % .6f % .6f]\n", source_point_.m_storage.m_data.array[0], source_point_.m_storage.m_data.array[1], source_point_.m_storage.m_data.array[2]
    printf "y_n (target) = [% .6f % .6f % .6f]\n", target_point_.m_storage.m_data.array[0], target_point_.m_storage.m_data.array[1], target_point_.m_storage.m_data.array[2]
    printf "n_y (normal) = [% .6f % .6f % .6f]\n", target_normal_.m_storage.m_data.array[0], target_normal_.m_storage.m_data.array[1], target_normal_.m_storage.m_data.array[2]
    printf "xi           = [% .6f % .6f % .6f % .6f % .6f % .6f]\n", parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]
    printf "\nresiduals[0] = % .12f   (1 value: r = n^T e)\n", residuals[0]
    printf "\njacobians[0] = 1x6, row major\n"
    printf "   idx        0         1         2         3         4         5\n"
    printf "   name      d/dtx     d/dty     d/dtz     d/da      d/db      d/dg\n"
    printf "   value % 9.5f % 9.5f % 9.5f % 9.5f % 9.5f % 9.5f\n", jacobians[0][0], jacobians[0][1], jacobians[0][2], jacobians[0][3], jacobians[0][4], jacobians[0][5]
    printf "\n   raw array: "
    print *jacobians[0]@6
    printf "   the first three are exactly n_y: the normal collapses de/dt = I_3 into one row.\n"
end
document jac
Dump the inputs, residual and Jacobian of the correspondence currently stopped in.
end

define nojac
    delete breakpoints
    printf "breakpoints cleared -- continuing now runs to completion\n"
end
document nojac
Drop every breakpoint and let the program run freely.
end
