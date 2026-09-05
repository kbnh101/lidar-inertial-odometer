# Terminal entry point for inspecting PointToPlaneCostFunction::Evaluate().
#
#   cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug -j$(nproc)
#   gdb -x debug.gdb ./build-debug/test_icp
#
# In VSCode press F5 instead; launch.json sources debug-commands.gdb and you set breakpoints in the
# gutter. Release builds inline Evaluate(), so a Debug build is required either way.

set pagination off
source debug-commands.gdb

# Line 84 is `return true;`, so residuals and jacobians are both already written.
break point_to_plane_cost.hpp:84
commands
    silent
    jac
end

printf "\nrun    -> stops at the first correspondence and prints it\n"
printf "c      -> next correspondence\n"
printf "jac    -> re-print the current one\n"
printf "nojac  -> stop breaking, run to the end\n\n"
