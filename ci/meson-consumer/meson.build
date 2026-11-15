project(
    'rgsaddle-consumer',
    'cpp',
    meson_version: '>=0.63.0',
    default_options: ['cpp_std=c++17', 'warning_level=3', 'werror=true'],
)

# gpridge's call, unchanged from the packagefiles overlay.
rgsaddle_dep = dependency(
    'rgsaddle-headers',
    fallback: ['rgsaddle-headers', 'rgsaddle_headers_dep'],
)

expect = get_option('expect')
if rgsaddle_dep.type_name() != expect
    error('rgsaddle-headers came from @0@, expected @1@'.format(
        rgsaddle_dep.type_name(),
        expect,
    ))
endif
if rgsaddle_dep.version() != get_option('rgsaddle_version')
    error('rgsaddle-headers is @0@, expected @1@'.format(
        rgsaddle_dep.version(),
        get_option('rgsaddle_version'),
    ))
endif

# As a subproject rgsaddle also answers to its own name.
if expect == 'internal'
    assert(dependency('rgsaddle', allow_fallback: false).type_name() == 'internal')
endif

exe = executable(
    'band',
    'band.cpp',
    dependencies: [rgsaddle_dep, dependency('dl')],
)
test('band', exe)
