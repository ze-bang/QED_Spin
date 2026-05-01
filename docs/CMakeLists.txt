# =============================================================================
# docs/CMakeLists.txt -- Doxygen + Sphinx + Breathe documentation site.
#
# Two opt-in targets, both gated on ED_BUILD_DOCS=ON in the parent build:
#
#   * `doxygen`  -- runs Doxygen, dumps XML to build/docs/_doxygen/xml/
#   * `sphinx`   -- depends on `doxygen`, runs sphinx-build to produce
#                   the HTML site under build/docs/sphinx/.
#
# CI publishes build/docs/sphinx/ to GitHub Pages on every push to main
# (see .github/workflows/docs.yml).
# =============================================================================

if(NOT ED_BUILD_DOCS)
    return()
endif()

find_package(Doxygen REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ----- Doxygen --------------------------------------------------------------

set(DOXYGEN_OUT          "${CMAKE_CURRENT_BINARY_DIR}/_doxygen")
set(DOXYGEN_INPUT_DIRS   "${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src")
set(DOXYGEN_INPUT_FILE   "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYGEN_OUTPUT_FILE  "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file("${DOXYGEN_INPUT_FILE}" "${DOXYGEN_OUTPUT_FILE}" @ONLY)

add_custom_command(
    OUTPUT  "${DOXYGEN_OUT}/xml/index.xml"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUT}"
    COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYGEN_OUTPUT_FILE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Doxygen: extracting C++ API XML to ${DOXYGEN_OUT}/xml/"
    DEPENDS "${DOXYGEN_OUTPUT_FILE}"
    VERBATIM
)
add_custom_target(doxygen DEPENDS "${DOXYGEN_OUT}/xml/index.xml")

# ----- Sphinx ---------------------------------------------------------------

set(SPHINX_BUILD          "${CMAKE_CURRENT_BINARY_DIR}/sphinx")
set(SPHINX_SOURCE         "${CMAKE_CURRENT_SOURCE_DIR}")

# Allow either a system sphinx-build or the pip-installed one inside the
# project's virtualenv. ED_SPHINX_BUILD overrides the auto-detected path.
if(DEFINED ED_SPHINX_BUILD)
    set(_sphinx_cmd "${ED_SPHINX_BUILD}")
else()
    find_program(SPHINX_BUILD_EXECUTABLE sphinx-build REQUIRED)
    set(_sphinx_cmd "${SPHINX_BUILD_EXECUTABLE}")
endif()

add_custom_target(sphinx
    COMMAND ${CMAKE_COMMAND} -E env
            "ED_DOXYGEN_XML=${DOXYGEN_OUT}/xml"
            "ED_DOC_VERSION=${PROJECT_VERSION}"
            ${_sphinx_cmd}
                -b html
                --keep-going
                -j auto
                -d "${SPHINX_BUILD}/.doctrees"
                "${SPHINX_SOURCE}"
                "${SPHINX_BUILD}/html"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS doxygen
    COMMENT "Sphinx: rendering HTML to ${SPHINX_BUILD}/html/"
    VERBATIM
)

message(STATUS "Docs targets enabled: build with `cmake --build build --target sphinx`.")
message(STATUS "  Doxygen XML  -> ${DOXYGEN_OUT}/xml")
message(STATUS "  Sphinx HTML  -> ${SPHINX_BUILD}/html")
