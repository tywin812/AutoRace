# EREVAN AutoRace
A ROS2 metapackage that has a solution of EREVAN team for Autorace competition.

## Usage for EREVAN AutoRace

1. Install dependencies

    ```bash
    pip install ultralytics
    ```

2. Build the project

    ```bash
    colcon build
    ```

3. Source the workspace

    ```bash
    . ~/template_ws/install/setup.bash
    ```

4. Launch the simulation

    ```bash
    ros2 launch robot_bringup autorace_2025.launch.py
    ```

5. Run your own launch file that controls the robot

    ```bash
    ros2 launch autorace_core_erevan autorace_core.launch.py
    ```

6. Run the referee

    ```bash
    ros2 run referee_console mission_autorace_2025_referee
    ```