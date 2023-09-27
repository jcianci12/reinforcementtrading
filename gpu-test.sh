# Pull down the GPU test image:
sudo docker pull nricklin/ubuntu-gpu-test

# Run the test:
DOCKER_NVIDIA_DEVICES="--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm"
sudo docker run $DOCKER_NVIDIA_DEVICES nricklin/ubuntu-gpu-test