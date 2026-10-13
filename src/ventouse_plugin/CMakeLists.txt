cmake_minimum_required(VERSION 3.16)
project(ventouse_plugin)

find_package(ament_cmake REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

add_library(Ventouse SHARED Ventouse.cc)
set_property(TARGET Ventouse PROPERTY CXX_STANDARD 17)
target_link_libraries(Ventouse gz-plugin2::gz-plugin2 gz-sim8::gz-sim8)

install(TARGETS Ventouse LIBRARY DESTINATION lib)

# Ajoute install/ventouse_plugin/lib à GZ_SIM_SYSTEM_PLUGIN_PATH quand on source install/setup.bash
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()
