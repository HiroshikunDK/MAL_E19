IMAGE=tfdev11
D=mytfdev11

UID=$(shell id -u)
GID=$(shell id -g)

TENSORFLOW_VERSION=r1.11
BAZEL_VERSION=21

include ../make.mk

build: build_common

run:
	@ - docker rm $(D) 
	#docker run -d -it -p 49448:22 -v /home:/home -v /mnt/Dat:/mnt/Dat -v $(PWD)/opt:/opt -h $(D) --name $(D) $(IMAGE)
	#docker attach $(D) || true
	docker run -u $(UID):$(GID)  -it -w /mnt/TF/tensorflow -v /mnt/Dat:/mnt -e HOST_PERMS="$(UID):$(GID)" -h $(D) --name $(D) $(IMAGE) bash 
	@echo "DONE"

attach:
	docker attach $(D) || true

tfclone:
	mkdir -p TF
	cd TF && git clone https://github.com/tensorflow/tensorflow
	cd TF/tensorflow && git checkout $(TENSORFLOW_VERSION)

getana:
	cd Files && wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh

getbazel:
	wget https://github.com/bazelbuild/bazel/releases/download/0.$(BAZEL_VERSION).0/bazel-0.$(BAZEL_VERSION).0-installer-linux-x86_64.sh
	
clean:
	@ docker stop $(D) 2>/dev/null || true
	@ docker rm $(shell docker ps -a -q) 2>/dev/null || true 
	@ docker rmi $(shell docker images | grep "^<none>" | cut -b 40-65) 2>/dev/null || true 
	@ #docker rm `docker ps --no-trunc -aq` || echo "OK"
	@ docker system prune -f
	@ echo "cleaned.."  
