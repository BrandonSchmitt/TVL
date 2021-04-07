configure_lit_site_cfg(
        lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        lit.cfg.py
)

set(TVL_TEST_DEPENDS
        FileCheck count not
        tvl-opt
        tvl-translate
        )

add_lit_testsuite(check-tvl "Running the tvl regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TVL_TEST_DEPENDS}
        )
set_target_properties(check-tvl PROPERTIES FOLDER "Tests")

add_lit_testsuites(tvl . DEPENDS ${TVL_TEST_DEPENDS})
