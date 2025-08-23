# Development Log: Docker Learning Notes

## 1. What is Docker

* **Definition**: Docker is a containerization platform that packages applications and their dependencies into containers, allowing them to run consistently across different environments without worrying about system differences.
* **Analogy**: A container is like a lightweight "independent room" with its own furniture (environment, libraries, configurations) that doesn't interfere with other rooms.

---

## 2. Core Concepts of Docker

* **Image**: A template for containers, like a "mold for storage boxes". For example, `ros:noetic`.
* **Container**: A running instance of an image, like an "actual box".
* **Dockerfile**: A recipe that defines what environment and dependencies to install in the image.
* **Docker Hub**: An image repository where you can download pre-built images from others.

---

## 3. Differences Between Docker and Virtual Machines

* **Virtual Machines**: Require running a complete operating system, slow startup, high resource overhead.
* **Docker Containers**: Share the host kernel, only package necessary dependencies, lightweight, fast startup, can run multiple instances simultaneously.
* **Analogy**: Virtual machines are like complete apartments, containers are like study room seats - space-efficient and more flexible to use.

---

## 4. Environment Isolation

* **Within the same container**: Like an independent computer, environments inside may conflict (e.g., ROS1 and ROS2).
* **Between different containers**: Completely isolated, no interference - this is Docker's most recommended usage.
* **Use cases**:

  * Project A uses Python 3.8 + TensorFlow
  * Project B uses Python 3.11 + PyTorch
    Place them in separate containers, no mutual interference.

---

## 5. Container Lifecycle

* **Start a container**

  ```bash
  docker run -it ubuntu:20.04
  ```

  Launches a new container.

* **Exit container**

  * If the container's foreground main process is `bash` → exit = container stops.
  * If the container has background services (like `roscore`) → exit only disconnects terminal, container continues running.

* **Stop container**

  ```bash
  docker stop my_container
  ```

  All processes in the container terminate, status becomes `Exited`, but data remains.

* **Remove container**

  ```bash
  docker rm my_container
  ```

  Container is completely deleted, data is gone unless volumes are mounted.

---

## 6. Running Multiple Containers and Communication

* **Same-machine communication**: Containers join the same custom network, access each other by container name.

  ```bash
  docker network create robotnet
  docker run -d --name ros1 --network robotnet ros:noetic
  docker run -d --name ros2 --network robotnet ros:foxy
  ```

  In the `ros2` container, you can access the `ros1` container using `ros1:11311`.

* **External communication**: Expose services through port mapping.

  ```bash
  docker run -d -p 8000:8000 my_api
  ```

* **Data sharing**: Use volumes to mount directories.

  ```bash
  docker run -v $(pwd)/ws:/ws my_container
  ```

* **docker-compose management**:
  A single `docker-compose.yml` file can launch a group of containers, suitable for ROS1 + ROS2 + bridge combinations.

### Understanding YAML Files (.yml/.yaml)

**What is YAML?**
* **YAML** stands for "YAML Ain't Markup Language" (recursive acronym) or "Yet Another Markup Language"
* It's a **human-readable data serialization standard** used for configuration files and data exchange
* **Key characteristics**: Clean syntax, indentation-based structure, easy to read and write

**Why YAML is everywhere:**
1. **Configuration files**: Most modern tools use YAML for configs (Docker Compose, Kubernetes, CI/CD pipelines)
2. **Data exchange**: APIs, databases, and applications use it to transfer structured data
3. **Infrastructure as Code**: Cloud deployments, automation scripts
4. **Documentation**: Some static site generators use YAML for metadata

**YAML vs other formats:**
```yaml
# YAML - Clean and readable
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret123
```

```json
// JSON - More verbose
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "credentials": {
      "username": "admin", 
      "password": "secret123"
    }
  }
}
```

**Docker Compose example:**
```yaml
version: '3.8'
services:
  ros1:
    image: ros:noetic
    container_name: ros1_container
    networks:
      - robotnet
  ros2:
    image: ros:foxy
    container_name: ros2_container
    networks:
      - robotnet
networks:
  robotnet:
    driver: bridge
```

**Common YAML use cases in robotics:**
- ROS launch files (`.launch.yaml`)
- Robot configuration parameters
- CI/CD pipelines (GitHub Actions, GitLab CI)
- Kubernetes deployments for robot clusters
- Docker Compose for multi-container robot systems

---

## 7. Application in ROS Scenarios

* **Recommended practice**: Run ROS1 and ROS2 in separate containers to avoid environment conflicts.
* **Interoperability**: Use `ros1_bridge` as an intermediate container to connect both sides.
* **Multi-machine or cross-container communication**:

  * ROS1 commonly uses `network_mode: host` to avoid complex configuration.
  * ROS2 supports DDS multicast, containers in the same network segment can discover each other.

---

## 8. Common Commands Summary

```bash
# View containers
docker ps -a

# Enter a running container
docker exec -it <container_name> bash

# Start/stop/remove containers
docker start <container_name>
docker stop <container_name>
docker rm <container_name>

# Save modified container as new image
docker commit <container_ID> my_ros:noetic_custom
```

---

## 9. My Understanding Evolution

1. Initially thought Docker was just "packaging files".
2. Later understood it's actually a tool for **environment isolation**, more like a lightweight virtual machine.
3. Learned about container lifecycle, understood that "exit ≠ stop".
4. Discovered that multiple containers can interconnect, suitable for breaking down complex projects.
5. In ROS scenarios, best practice is **separating ROS1 and ROS2 into different containers** with bridge communication.

---

**Summary**: Docker's core value is *environment consistency* and *isolation*.
In my project, it ensures that ROS, Python, AI models and other modules run independently, then combine through network/volume integration.

---
