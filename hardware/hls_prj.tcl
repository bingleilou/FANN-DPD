open_project -reset hls_mlp_prj
set_top MLP
add_files MLP.cpp
add_files -tb tb.cpp

add_files -tb data
open_solution -reset "solution-rfsoc"
set_part {xczu28dr-ffvg1517-2-e}
    
create_clock -period 2.0 -name default
csim_design -clean
csynth_design
# cosim_design
export_design -format ip_catalog
exit