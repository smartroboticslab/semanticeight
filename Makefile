.PHONY: release
release:
	mkdir -p build/release
	cd build/release && cmake -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/release $(MFLAGS)

.PHONY: release-with-debug
release-with-debug:
	mkdir -p build/relwithdebinfo
	cd build/relwithdebinfo && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/relwithdebinfo $(MFLAGS)

.PHONY: debug
debug:
	mkdir -p build/debug/logs
	cd build/debug && cmake -DCMAKE_BUILD_TYPE=Debug $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/debug $(MFLAGS)

.PHONY: install
install:
	$(MAKE) -C build/release $(MFLAGS) install

.PHONY: uninstall
uninstall:
	$(MAKE) -C build/release $(MFLAGS) uninstall



.PHONY: build-tests
build-tests:
	$(MAKE) -C se_shared/test $(MFLAGS)
	$(MAKE) -C se_core/test $(MFLAGS)
	$(MAKE) -C se_voxel_impl/test $(MFLAGS)
	$(MAKE) -C se_denseslam/test $(MFLAGS)
	$(MAKE) -C se_apps/test $(MFLAGS)

.PHONY: test
test: build-tests
	$(MAKE) -C se_shared/test $(MFLAGS) test
	$(MAKE) -C se_core/test $(MFLAGS) test
	$(MAKE) -C se_voxel_impl/test $(MFLAGS) test
	$(MAKE) -C se_denseslam/test $(MFLAGS) test
	$(MAKE) -C se_apps/test $(MFLAGS) test

.PHONY: clean-tests
clean-tests:
	$(MAKE) -C se_shared/test $(MFLAGS) clean
	$(MAKE) -C se_core/test $(MFLAGS) clean
	$(MAKE) -C se_voxel_impl/test $(MFLAGS) clean
	$(MAKE) -C se_denseslam/test $(MFLAGS) clean
	$(MAKE) -C se_apps/test $(MFLAGS) clean



.PHONY: doc
doc:
	doxygen



.PHONY: clean
clean:
	rm -rf build

.PHONY: clean-doc
clean-doc:
	rm -rf doc/html

