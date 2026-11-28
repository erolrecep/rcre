/* SDL3 window + OpenGL 3.3 renderer.
 * YUV->RGB conversion happens in the fragment shader for quality/speed.
 * Supports NV12 (Y + interleaved UV planes) and RGB24 (decoded MJPEG).
 * Includes an interactive on-screen OpenGL GUI (Toolbar, Filters, Snapshot, Mirror, HUD). */
#include "render.h"
#include "font.h"

#include <SDL3/SDL.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif

#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
#endif
#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_TEXTURE0
#define GL_TEXTURE0 0x84C0
#endif
#ifndef GL_RG
#define GL_RG 0x8227
#endif
#ifndef GL_R8
#define GL_R8 0x8229
#endif
#ifndef GL_RG8
#define GL_RG8 0x822B
#endif
#ifndef GL_RED
#define GL_RED 0x1903
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_BLEND
#define GL_BLEND 0x0BE2
#endif
#ifndef GL_SRC_ALPHA
#define GL_SRC_ALPHA 0x0302
#endif
#ifndef GL_ONE_MINUS_SRC_ALPHA
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#endif
#ifndef GL_TRIANGLES
#define GL_TRIANGLES 0x0004
#endif

/* OpenGL 3.3 functions resolved at runtime via SDL_GL_GetProcAddress.
 * Own typedefs so we don't depend on glext.h (absent/limited on macOS). */
#ifndef APIENTRY
#define APIENTRY
#endif
#define GLPROC(ret, name, args) \
    typedef ret (APIENTRY *name##_fn) args; static name##_fn p##name
GLPROC(GLuint, glCreateShader, (GLenum));
GLPROC(void, glShaderSource, (GLuint, GLsizei, const char *const *, const GLint *));
GLPROC(void, glCompileShader, (GLuint));
GLPROC(void, glGetShaderiv, (GLuint, GLenum, GLint *));
GLPROC(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei *, char *));
GLPROC(GLuint, glCreateProgram, (void));
GLPROC(void, glAttachShader, (GLuint, GLuint));
GLPROC(void, glLinkProgram, (GLuint));
GLPROC(void, glDeleteShader, (GLuint));
GLPROC(void, glDeleteProgram, (GLuint));
GLPROC(void, glUseProgram, (GLuint));
GLPROC(GLint, glGetUniformLocation, (GLuint, const char *));
GLPROC(void, glUniform1i, (GLint, GLint));
GLPROC(void, glUniform1f, (GLint, GLfloat));
GLPROC(void, glUniform2f, (GLint, GLfloat, GLfloat));
GLPROC(void, glGenVertexArrays, (GLsizei, GLuint *));
GLPROC(void, glBindVertexArray, (GLuint));
GLPROC(void, glGenBuffers, (GLsizei, GLuint *));
GLPROC(void, glBindBuffer, (GLenum, GLuint));
GLPROC(void, glBufferData, (GLenum, ptrdiff_t, const void *, GLenum));
GLPROC(void, glEnableVertexAttribArray, (GLuint));
GLPROC(void, glVertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void *));
GLPROC(void, glActiveTexture, (GLenum));

static int load_gl_procs(void)
{
#define LOAD(name) \
    p##name = (name##_fn)SDL_GL_GetProcAddress(#name); \
    if (!p##name) { fprintf(stderr, "missing GL proc: %s\n", #name); return 0; }
    LOAD(glCreateShader); LOAD(glShaderSource); LOAD(glCompileShader);
    LOAD(glGetShaderiv); LOAD(glGetShaderInfoLog); LOAD(glCreateProgram);
    LOAD(glAttachShader); LOAD(glLinkProgram); LOAD(glDeleteShader);
    LOAD(glDeleteProgram); LOAD(glUseProgram); LOAD(glGetUniformLocation);
    LOAD(glUniform1i); LOAD(glUniform1f); LOAD(glUniform2f);
    LOAD(glGenVertexArrays); LOAD(glBindVertexArray); LOAD(glGenBuffers);
    LOAD(glBindBuffer); LOAD(glBufferData); LOAD(glEnableVertexAttribArray);
    LOAD(glVertexAttribPointer); LOAD(glActiveTexture);
#undef LOAD
    return 1;
}

#define glCreateShader pglCreateShader
#define glShaderSource pglShaderSource
#define glCompileShader pglCompileShader
#define glGetShaderiv pglGetShaderiv
#define glGetShaderInfoLog pglGetShaderInfoLog
#define glCreateProgram pglCreateProgram
#define glAttachShader pglAttachShader
#define glLinkProgram pglLinkProgram
#define glDeleteShader pglDeleteShader
#define glDeleteProgram pglDeleteProgram
#define glUseProgram pglUseProgram
#define glGetUniformLocation pglGetUniformLocation
#define glUniform1i pglUniform1i
#define glUniform1f pglUniform1f
#define glUniform2f pglUniform2f
#define glGenVertexArrays pglGenVertexArrays
#define glBindVertexArray pglBindVertexArray
#define glGenBuffers pglGenBuffers
#define glBindBuffer pglBindBuffer
#define glBufferData pglBufferData
#define glEnableVertexAttribArray pglEnableVertexAttribArray
#define glVertexAttribPointer pglVertexAttribPointer
#define glActiveTexture pglActiveTexture

#define MAX_OVERLAY_VERTS 4096

enum {
    ACTION_MIRROR = 1,
    ACTION_PAUSE,
    ACTION_SNAPSHOT,
    ACTION_FILTER,
    ACTION_STATS
};

struct renderer {
    SDL_Window   *win;
    SDL_GLContext gl;
    GLuint        prog_nv12;
    GLuint        prog_rgb;
    GLuint        prog_overlay;
    GLuint        tex_y, tex_uv; /* NV12 path; RGB24 uses tex_y only */
    GLuint        tex_font;
    GLuint        vao, vbo;
    GLuint        overlay_vao, overlay_vbo;

    /* Overlay vertex batch: (x, y, u, v, r, g, b, a) */
    float         verts[MAX_OVERLAY_VERTS * 8];
    int           nverts;

    /* Live stats */
    uint64_t      last_stat_update_ns;
    uint32_t      frames_since_update;
    float         current_fps;
    float         current_frame_ms;
    int           stats_visible;

    /* Interactive GUI states */
    int           mirror;
    int           paused;
    int           filter_mode; /* 0: Normal, 1: Grayscale, 2: Contrast, 3: Sepia, 4: Invert */
    float         flash;
    int           snapshot_requested;

    /* Mouse input & Toolbar state */
    float         mouse_win_x;
    float         mouse_win_y;
    int           mouse_clicked;
    uint64_t      last_mouse_move_ns;

    /* Toast notification */
    char          toast_text[128];
    uint64_t      toast_expire_ns;
};

static renderer_t *g_current_renderer = NULL;

static const char *vs_src =
    "#version 330 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=1) in vec2 uv_in;\n"
    "uniform int u_mirror;\n"
    "out vec2 uv;\n"
    "void main(){\n"
    "  vec2 u = uv_in;\n"
    "  if (u_mirror == 1) u.x = 1.0 - u.x;\n"
    "  uv = u;\n"
    "  gl_Position = vec4(pos, 0.0, 1.0);\n"
    "}\n";

/* NV12: sample Y plane + interleaved UV plane, convert BT.601 with filters. */
static const char *fs_nv12_src =
    "#version 330 core\n"
    "in vec2 uv;\n"
    "out vec4 frag;\n"
    "uniform sampler2D tex_y;\n"
    "uniform sampler2D tex_uv;\n"
    "uniform int u_filter;\n"
    "uniform float u_flash;\n"
    "void main(){\n"
    "  float y = texture(tex_y, uv).r;\n"
    "  vec2 cbcr = texture(tex_uv, uv).rg - 0.5;\n"
    "  float r = y + 1.402 * cbcr.y;\n"
    "  float g = y - 0.344136 * cbcr.x - 0.714136 * cbcr.y;\n"
    "  float b = y + 1.772 * cbcr.x;\n"
    "  vec3 col = clamp(vec3(r, g, b), 0.0, 1.0);\n"
    "  if (u_filter == 1) {\n"
    "    float gray = dot(col, vec3(0.299, 0.587, 0.114));\n"
    "    col = vec3(gray);\n"
    "  } else if (u_filter == 2) {\n"
    "    col = clamp((col - 0.5) * 1.5 + 0.5, 0.0, 1.0);\n"
    "  } else if (u_filter == 3) {\n"
    "    col = clamp(vec3(\n"
    "      dot(col, vec3(0.393, 0.769, 0.189)),\n"
    "      dot(col, vec3(0.349, 0.686, 0.168)),\n"
    "      dot(col, vec3(0.272, 0.534, 0.131))\n"
    "    ), 0.0, 1.0);\n"
    "  } else if (u_filter == 4) {\n"
    "    col = 1.0 - col;\n"
    "  }\n"
    "  if (u_flash > 0.0) {\n"
    "    col = mix(col, vec3(1.0), u_flash);\n"
    "  }\n"
    "  frag = vec4(col, 1.0);\n"
    "}\n";

static const char *fs_rgb_src =
    "#version 330 core\n"
    "in vec2 uv;\n"
    "out vec4 frag;\n"
    "uniform sampler2D tex_y;\n"
    "uniform int u_filter;\n"
    "uniform float u_flash;\n"
    "void main(){\n"
    "  vec3 col = texture(tex_y, uv).rgb;\n"
    "  if (u_filter == 1) {\n"
    "    float gray = dot(col, vec3(0.299, 0.587, 0.114));\n"
    "    col = vec3(gray);\n"
    "  } else if (u_filter == 2) {\n"
    "    col = clamp((col - 0.5) * 1.5 + 0.5, 0.0, 1.0);\n"
    "  } else if (u_filter == 3) {\n"
    "    col = clamp(vec3(\n"
    "      dot(col, vec3(0.393, 0.769, 0.189)),\n"
    "      dot(col, vec3(0.349, 0.686, 0.168)),\n"
    "      dot(col, vec3(0.272, 0.534, 0.131))\n"
    "    ), 0.0, 1.0);\n"
    "  } else if (u_filter == 4) {\n"
    "    col = 1.0 - col;\n"
    "  }\n"
    "  if (u_flash > 0.0) {\n"
    "    col = mix(col, vec3(1.0), u_flash);\n"
    "  }\n"
    "  frag = vec4(col, 1.0);\n"
    "}\n";

static const char *vs_overlay_src =
    "#version 330 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=1) in vec2 uv_in;\n"
    "layout(location=2) in vec4 color_in;\n"
    "uniform vec2 u_screen;\n"
    "out vec2 uv;\n"
    "out vec4 color;\n"
    "void main(){\n"
    "  vec2 ndc = (pos / u_screen) * 2.0 - 1.0;\n"
    "  ndc.y = -ndc.y;\n"
    "  gl_Position = vec4(ndc, 0.0, 1.0);\n"
    "  uv = uv_in;\n"
    "  color = color_in;\n"
    "}\n";

static const char *fs_overlay_src =
    "#version 330 core\n"
    "in vec2 uv;\n"
    "in vec4 color;\n"
    "out vec4 frag;\n"
    "uniform sampler2D u_font;\n"
    "void main(){\n"
    "  float a = texture(u_font, uv).r;\n"
    "  frag = vec4(color.rgb, color.a * a);\n"
    "}\n";

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "shader: %s\n", log);
    }
    return s;
}

static GLuint link_program_custom(const char *vs_code, const char *fs_code)
{
    GLuint vs = compile(GL_VERTEX_SHADER, vs_code);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fs_code);
    GLuint p  = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

static GLuint link_program(const char *fs_src)
{
    return link_program_custom(vs_src, fs_src);
}

static void add_quad(renderer_t *r, float x, float y, float w, float h,
                     float u0, float v0, float u1, float v1,
                     float cr, float cg, float cb, float ca)
{
    if (r->nverts + 6 > MAX_OVERLAY_VERTS)
        return;

    float *v = &r->verts[r->nverts * 8];

    /* Triangle 1 */
    v[0] = x;     v[1] = y;     v[2] = u0; v[3] = v0; v[4] = cr; v[5] = cg; v[6] = cb; v[7] = ca;
    v[8] = x + w; v[9] = y;     v[10]= u1; v[11]= v0; v[12]= cr; v[13]= cg; v[14]= cb; v[15]= ca;
    v[16]= x;     v[17]= y + h; v[18]= u0; v[19]= v1; v[20]= cr; v[21]= cg; v[22]= cb; v[23]= ca;

    /* Triangle 2 */
    v[24]= x + w; v[25]= y;     v[26]= u1; v[27]= v0; v[28]= cr; v[29]= cg; v[30]= cb; v[31]= ca;
    v[32]= x + w; v[33]= y + h; v[34]= u1; v[35]= v1; v[36]= cr; v[37]= cg; v[38]= cb; v[39]= ca;
    v[40]= x;     v[41]= y + h; v[42]= u0; v[43]= v1; v[44]= cr; v[45]= cg; v[46]= cb; v[47]= ca;

    r->nverts += 6;
}

static void add_solid_rect(renderer_t *r, float x, float y, float w, float h,
                           float cr, float cg, float cb, float ca)
{
    float u = 8.0f / (float)FONT_ATLAS_W;
    float v = 8.0f / (float)FONT_ATLAS_H;
    add_quad(r, x, y, w, h, u, v, u, v, cr, cg, cb, ca);
}

static void add_char(renderer_t *r, char c, float x, float y, float w, float h,
                     float cr, float cg, float cb, float ca)
{
    if ((unsigned char)c < FONT_FIRST_CHAR || (unsigned char)c >= FONT_FIRST_CHAR + 95)
        return;
    int idx = (unsigned char)c - FONT_FIRST_CHAR;
    int col = idx % FONT_COLS;
    int row = idx / FONT_COLS;
    float u0 = (float)(col * FONT_CELL_W) / (float)FONT_ATLAS_W;
    float v0 = (float)(row * FONT_CELL_H) / (float)FONT_ATLAS_H;
    float u1 = (float)((col + 1) * FONT_CELL_W) / (float)FONT_ATLAS_W;
    float v1 = (float)((row + 1) * FONT_CELL_H) / (float)FONT_ATLAS_H;

    add_quad(r, x, y, w, h, u0, v0, u1, v1, cr, cg, cb, ca);
}

static void add_string(renderer_t *r, const char *str, float x, float y,
                       float cw, float ch, float cr, float cg, float cb, float ca)
{
    float cur_x = x;
    for (const char *p = str; *p; p++) {
        if (*p != ' ')
            add_char(r, *p, cur_x, y, cw, ch, cr, cg, cb, ca);
        cur_x += cw;
    }
}

static void trigger_action(renderer_t *r, int action)
{
    uint64_t now = SDL_GetTicksNS();
    switch (action) {
    case ACTION_MIRROR:
        r->mirror = !r->mirror;
        snprintf(r->toast_text, sizeof(r->toast_text), "Mirror %s", r->mirror ? "ON" : "OFF");
        r->toast_expire_ns = now + 1800000000ULL;
        break;
    case ACTION_PAUSE:
        r->paused = !r->paused;
        snprintf(r->toast_text, sizeof(r->toast_text), "Stream %s", r->paused ? "PAUSED" : "RESUMED");
        r->toast_expire_ns = now + 1800000000ULL;
        break;
    case ACTION_SNAPSHOT:
        r->snapshot_requested = 1;
        break;
    case ACTION_FILTER: {
        r->filter_mode = (r->filter_mode + 1) % 5;
        const char *names[] = { "Normal", "Grayscale", "Contrast", "Sepia", "Invert" };
        snprintf(r->toast_text, sizeof(r->toast_text), "Filter: %s", names[r->filter_mode]);
        r->toast_expire_ns = now + 1800000000ULL;
        break;
    }
    case ACTION_STATS:
        r->stats_visible = !r->stats_visible;
        snprintf(r->toast_text, sizeof(r->toast_text), "HUD %s", r->stats_visible ? "ON" : "OFF");
        r->toast_expire_ns = now + 1800000000ULL;
        break;
    }
}

static void take_snapshot(renderer_t *r, int fb_w, int fb_h)
{
    r->flash = 0.85f; /* Trigger camera shutter flash */

    /* Ensure snapshots folder exists */
#ifdef _WIN32
    _mkdir("snapshots");
#else
    mkdir("snapshots", 0755);
#endif

    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char filename[160];
    strftime(filename, sizeof(filename), "snapshots/webcam-snapshot-%Y%m%d-%H%M%S.bmp", tm_info);

    uint32_t *raw = malloc((size_t)fb_w * fb_h * 4);
    uint32_t *flipped = malloc((size_t)fb_w * fb_h * 4);
    if (!raw || !flipped) {
        free(raw);
        free(flipped);
        return;
    }

    glReadPixels(0, 0, fb_w, fb_h, GL_RGBA, GL_UNSIGNED_BYTE, raw);
    for (int y = 0; y < fb_h; y++) {
        memcpy(&flipped[y * fb_w], &raw[(fb_h - 1 - y) * fb_w], (size_t)fb_w * 4);
    }

    SDL_Surface *surf = SDL_CreateSurfaceFrom(fb_w, fb_h, SDL_PIXELFORMAT_RGBA32, flipped, fb_w * 4);
    if (surf) {
        SDL_SaveBMP(surf, filename);
        SDL_DestroySurface(surf);
        snprintf(r->toast_text, sizeof(r->toast_text), "Saved %s", filename);
        r->toast_expire_ns = SDL_GetTicksNS() + 3000000000ULL;
    }

    free(raw);
    free(flipped);
}

renderer_t *render_create(int width, int height, const char *title)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return NULL;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);

    renderer_t *r = calloc(1, sizeof(*r));
    r->win = SDL_CreateWindow(title, width, height,
                              SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!r->win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        free(r);
        return NULL;
    }
    r->gl = SDL_GL_CreateContext(r->win);
    if (!r->gl || !load_gl_procs()) {
        fprintf(stderr, "OpenGL 3.3 context unavailable: %s\n", SDL_GetError());
        SDL_DestroyWindow(r->win);
        free(r);
        return NULL;
    }
    SDL_GL_SetSwapInterval(1); /* vsync */

    r->prog_nv12    = link_program(fs_nv12_src);
    r->prog_rgb     = link_program(fs_rgb_src);
    r->prog_overlay = link_program_custom(vs_overlay_src, fs_overlay_src);

    /* Fullscreen triangle strip for video frame. */
    static const float quad[] = {
        /* pos      uv */
        -1.f, -1.f, 0.f, 1.f,
         1.f, -1.f, 1.f, 1.f,
        -1.f,  1.f, 0.f, 0.f,
         1.f,  1.f, 1.f, 0.f,
    };
    glGenVertexArrays(1, &r->vao);
    glGenBuffers(1, &r->vbo);
    glBindVertexArray(r->vao);
    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)(2 * sizeof(float)));

    glGenTextures(1, &r->tex_y);
    glGenTextures(1, &r->tex_uv);

    /* Overlay VAO & dynamic VBO */
    glGenVertexArrays(1, &r->overlay_vao);
    glGenBuffers(1, &r->overlay_vbo);
    glBindVertexArray(r->overlay_vao);
    glBindBuffer(GL_ARRAY_BUFFER, r->overlay_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));

    /* Font atlas texture (Dedicated Menlo) */
    glGenTextures(1, &r->tex_font);
    glBindTexture(GL_TEXTURE_2D, r->tex_font);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, FONT_ATLAS_W, FONT_ATLAS_H, 0,
                 GL_RED, GL_UNSIGNED_BYTE, font_atlas_menlo);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    r->stats_visible = 1;
    r->mirror = 0;
    r->paused = 0;
    r->filter_mode = 0;
    r->flash = 0.0f;
    r->last_mouse_move_ns = SDL_GetTicksNS();
    g_current_renderer = r;

    return r;
}

static void upload_nv12(renderer_t *r, const frame_t *f)
{
    glUseProgram(r->prog_nv12);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, r->tex_y);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, f->stride);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, f->width, f->height, 0,
                 GL_RED, GL_UNSIGNED_BYTE, f->data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, r->tex_uv);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, f->stride / 2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, f->width / 2, f->height / 2, 0,
                 GL_RG, GL_UNSIGNED_BYTE,
                 f->data + (size_t)f->stride * f->height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glUniform1i(glGetUniformLocation(r->prog_nv12, "tex_y"), 0);
    glUniform1i(glGetUniformLocation(r->prog_nv12, "tex_uv"), 1);
    glUniform1i(glGetUniformLocation(r->prog_nv12, "u_mirror"), r->mirror);
    glUniform1i(glGetUniformLocation(r->prog_nv12, "u_filter"), r->filter_mode);
    glUniform1f(glGetUniformLocation(r->prog_nv12, "u_flash"), r->flash);
}

static void upload_rgb(renderer_t *r, const frame_t *f)
{
    glUseProgram(r->prog_rgb);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, r->tex_y);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, f->stride / 3);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, f->width, f->height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, f->data);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glUniform1i(glGetUniformLocation(r->prog_rgb, "tex_y"), 0);
    glUniform1i(glGetUniformLocation(r->prog_rgb, "u_mirror"), r->mirror);
    glUniform1i(glGetUniformLocation(r->prog_rgb, "u_filter"), r->filter_mode);
    glUniform1f(glGetUniformLocation(r->prog_rgb, "u_flash"), r->flash);
}

static const char *get_aspect_ratio_str(int w, int h)
{
    if (w <= 0 || h <= 0) return "";
    if (w * 9 == h * 16) return "16:9";
    if (w * 3 == h * 4)  return "4:3";
    if (w * 10 == h * 16) return "16:10";
    if (w * 2 == h * 3)  return "3:2";
    if (w == h)          return "1:1";
    return "Custom";
}

static void render_stats_overlay(renderer_t *r, const frame_t *f,
                                 int fb_w, int fb_h, int win_w, int win_h)
{
    (void)fb_h;
    uint64_t now_ns = SDL_GetTicksNS();
    if (r->last_stat_update_ns == 0) {
        r->last_stat_update_ns = now_ns;
        r->current_fps = 0.0f;
        r->current_frame_ms = 0.0f;
    }
    r->frames_since_update++;
    uint64_t elapsed_ns = now_ns - r->last_stat_update_ns;
    if (elapsed_ns >= 400000000ULL) { /* Refresh stats every 400ms */
        r->current_fps = (float)r->frames_since_update * 1e9f / (float)elapsed_ns;
        r->current_frame_ms = (float)elapsed_ns / (float)r->frames_since_update / 1e6f;
        r->frames_since_update = 0;
        r->last_stat_update_ns = now_ns;

        char title[128];
        snprintf(title, sizeof(title), "webcam-stream — %.1f FPS (%dx%d)%s%s",
                 r->current_fps, f->width, f->height,
                 r->paused ? " [PAUSED]" : "",
                 r->mirror ? " [MIRROR]" : "");
        SDL_SetWindowTitle(r->win, title);
    }

    if (!r->stats_visible)
        return;

    float dpi = (win_w > 0) ? ((float)fb_w / (float)win_w) : 1.0f;
    if (dpi < 1.0f) dpi = 1.0f;

    float char_w  = 9.5f * dpi;
    float char_h  = 19.0f * dpi;
    float pad_x   = 12.0f * dpi;
    float pad_y   = 10.0f * dpi;
    float gap_y   = 4.0f * dpi;
    float margin  = 16.0f * dpi;
    float dot_sz  = 7.0f * dpi;
    float dot_gap = 8.0f * dpi;

    /* Line 1: Status & Performance */
    char l1_status[16];
    char l1_fps[32];
    char l1_ms[32];
    if (r->paused) {
        snprintf(l1_status, sizeof(l1_status), "PAUSED");
        snprintf(l1_fps, sizeof(l1_fps), "--.- FPS");
        snprintf(l1_ms, sizeof(l1_ms), "(Frozen)");
    } else {
        snprintf(l1_status, sizeof(l1_status), "LIVE");
        snprintf(l1_fps, sizeof(l1_fps), "%.1f FPS", r->current_fps);
        snprintf(l1_ms, sizeof(l1_ms), "%.1f ms", r->current_frame_ms);
    }

    /* Line 2: Capture Details (Resolution, Aspect Ratio, Pixel Format, Bandwidth) */
    const char *fmt_name = "NV12";
    if (f->fmt == PIXFMT_RGB24) fmt_name = "RGB24";
    else if (f->fmt == PIXFMT_YUYV) fmt_name = "YUYV";
    else if (f->fmt == PIXFMT_MJPEG) fmt_name = "MJPEG";

    const char *aspect = get_aspect_ratio_str(f->width, f->height);

    float bytes_per_frame = 0.0f;
    if (f->fmt == PIXFMT_RGB24)
        bytes_per_frame = (float)f->width * f->height * 3.0f;
    else if (f->fmt == PIXFMT_YUYV)
        bytes_per_frame = (float)f->width * f->height * 2.0f;
    else /* NV12 */
        bytes_per_frame = (float)f->width * f->height * 1.5f;

    float mb_s = r->paused ? 0.0f : (bytes_per_frame * r->current_fps) / (1024.0f * 1024.0f);

    char l2_cam[64];
    snprintf(l2_cam, sizeof(l2_cam), "%dx%d (%s)  %s  %.1f MB/s",
             f->width, f->height, aspect, fmt_name, mb_s);

    /* Line 3: Filter Info */
    const char *filter_names[] = {
        "Normal (Pass-through)",
        "Grayscale (BT.601)",
        "High Contrast (1.5x)",
        "Warm Sepia Tone",
        "Color Invert (Negative)"
    };
    char l3_fx[64];
    snprintf(l3_fx, sizeof(l3_fx), "%s", filter_names[r->filter_mode]);

    /* Line 4: Display View & Window Info */
    char l4_view[64];
    snprintf(l4_view, sizeof(l4_view), "%s  |  Win: %dx%d",
             r->mirror ? "Mirror: ON (Selfie)" : "Mirror: OFF",
             win_w, win_h);

    /* Layout metrics */
    float l1_w = dot_sz + dot_gap + ((float)strlen(l1_status) + 2.0f + (float)strlen(l1_fps) + 2.0f + (float)strlen(l1_ms)) * char_w;
    float l2_w = (6.0f + (float)strlen(l2_cam)) * char_w;
    float l3_w = (6.0f + (float)strlen(l3_fx)) * char_w;
    float l4_w = (6.0f + (float)strlen(l4_view)) * char_w;

    float content_w = l1_w;
    if (l2_w > content_w) content_w = l2_w;
    if (l3_w > content_w) content_w = l3_w;
    if (l4_w > content_w) content_w = l4_w;

    float box_w = content_w + pad_x * 2.0f;
    float box_h = char_h * 4.0f + gap_y * 3.0f + pad_y * 2.0f + 6.0f * dpi;

    float box_x = (float)fb_w - margin - box_w;
    float box_y = margin;

    /* 1. Backdrop translucent card */
    add_solid_rect(r, box_x, box_y, box_w, box_h, 0.04f, 0.06f, 0.09f, 0.86f);

    /* 2. Sleek border */
    float border_t = 1.0f * dpi;
    add_solid_rect(r, box_x, box_y, box_w, border_t, 0.24f, 0.30f, 0.40f, 0.75f);
    add_solid_rect(r, box_x, box_y + box_h - border_t, box_w, border_t, 0.24f, 0.30f, 0.40f, 0.75f);
    add_solid_rect(r, box_x, box_y, border_t, box_h, 0.24f, 0.30f, 0.40f, 0.75f);
    add_solid_rect(r, box_x + box_w - border_t, box_y, border_t, box_h, 0.24f, 0.30f, 0.40f, 0.75f);

    /* 3. Divider line between Line 1 and details */
    float div_y = box_y + pad_y + char_h + gap_y * 0.75f;
    add_solid_rect(r, box_x + pad_x * 0.5f, div_y, box_w - pad_x, border_t, 0.18f, 0.23f, 0.32f, 0.60f);

    /* 4. Line 1: Status Dot & Texts */
    float dot_x = box_x + pad_x;
    float dot_y = box_y + pad_y + (char_h - dot_sz) * 0.5f;
    if (r->paused)
        add_solid_rect(r, dot_x, dot_y, dot_sz, dot_sz, 0.95f, 0.75f, 0.20f, 1.0f); /* Amber */
    else
        add_solid_rect(r, dot_x, dot_y, dot_sz, dot_sz, 0.20f, 0.90f, 0.40f, 1.0f); /* Green */

    float text_x = dot_x + dot_sz + dot_gap;
    float text_y1 = box_y + pad_y;

    /* Status word (LIVE / PAUSED) */
    if (r->paused)
        add_string(r, l1_status, text_x, text_y1, char_w, char_h, 0.95f, 0.75f, 0.20f, 1.0f);
    else
        add_string(r, l1_status, text_x, text_y1, char_w, char_h, 0.25f, 0.90f, 0.45f, 1.0f);

    /* FPS text */
    float fps_x = text_x + ((float)strlen(l1_status) + 2.0f) * char_w;
    add_string(r, l1_fps, fps_x, text_y1, char_w, char_h, 1.0f, 1.0f, 1.0f, 1.0f);

    /* Frame latency */
    float ms_x = fps_x + ((float)strlen(l1_fps) + 2.0f) * char_w;
    add_string(r, l1_ms, ms_x, text_y1, char_w, char_h, 0.55f, 0.80f, 0.95f, 0.90f);

    /* 5. Line 2: Capture Info */
    float text_y2 = text_y1 + char_h + gap_y + 4.0f * dpi;
    add_string(r, "CAP   ", dot_x, text_y2, char_w, char_h, 0.45f, 0.60f, 0.78f, 0.85f);
    add_string(r, l2_cam, dot_x + 6.0f * char_w, text_y2, char_w, char_h, 0.90f, 0.92f, 0.96f, 0.95f);

    /* 6. Line 3: Filter Info */
    float text_y3 = text_y2 + char_h + gap_y;
    add_string(r, "FX    ", dot_x, text_y3, char_w, char_h, 0.45f, 0.60f, 0.78f, 0.85f);
    if (r->filter_mode != 0)
        add_string(r, l3_fx, dot_x + 6.0f * char_w, text_y3, char_w, char_h, 1.0f, 0.82f, 0.35f, 1.0f); /* Amber highlight */
    else
        add_string(r, l3_fx, dot_x + 6.0f * char_w, text_y3, char_w, char_h, 0.80f, 0.84f, 0.90f, 0.85f);

    /* 7. Line 4: View & Window Info */
    float text_y4 = text_y3 + char_h + gap_y;
    add_string(r, "VIEW  ", dot_x, text_y4, char_w, char_h, 0.45f, 0.60f, 0.78f, 0.85f);
    add_string(r, l4_view, dot_x + 6.0f * char_w, text_y4, char_w, char_h,
               r->mirror ? 0.35f : 0.75f,
               r->mirror ? 0.92f : 0.78f,
               r->mirror ? 0.65f : 0.84f, 0.90f);
}

static void render_gui_toolbar(renderer_t *r, int fb_w, int fb_h, int win_w)
{
    float dpi = (win_w > 0) ? ((float)fb_w / (float)win_w) : 1.0f;
    if (dpi < 1.0f) dpi = 1.0f;

    uint64_t now_ns = SDL_GetTicksNS();
    float mouse_fb_x = r->mouse_win_x * dpi;
    float mouse_fb_y = r->mouse_win_y * dpi;

    /* Toolbar buttons definition */
    struct {
        int action;
        const char *label;
        int active;
    } buttons[] = {
        { ACTION_MIRROR,   r->mirror ? "Mirror: ON" : "Mirror: OFF", r->mirror },
        { ACTION_PAUSE,    r->paused ? "Resume" : "Pause",           r->paused },
        { ACTION_SNAPSHOT, "Snapshot",                               0 },
        { ACTION_FILTER,
          (r->filter_mode == 0) ? "Filter: Normal" :
          (r->filter_mode == 1) ? "Filter: Mono" :
          (r->filter_mode == 2) ? "Filter: Contrast" :
          (r->filter_mode == 3) ? "Filter: Sepia" : "Filter: Invert",
          r->filter_mode != 0 },
        { ACTION_STATS,    r->stats_visible ? "HUD: ON" : "HUD: OFF", r->stats_visible }
    };
    int num_buttons = (int)(sizeof(buttons) / sizeof(buttons[0]));

    float char_w   = 9.5f * dpi;
    float char_h   = 19.0f * dpi;
    float pad_btn_x= 10.0f * dpi;
    float pad_btn_y= 8.0f * dpi;
    float btn_h    = char_h + pad_btn_y * 2.0f;
    float spacing  = 8.0f * dpi;
    float pad_bar_x= 10.0f * dpi;
    float pad_bar_y= 8.0f * dpi;

    /* Calculate total width */
    float btn_widths[8];
    float total_btns_w = 0.0f;
    for (int i = 0; i < num_buttons; i++) {
        btn_widths[i] = (float)strlen(buttons[i].label) * char_w + pad_btn_x * 2.0f;
        total_btns_w += btn_widths[i];
    }
    total_btns_w += spacing * (num_buttons - 1);

    float bar_w = total_btns_w + pad_bar_x * 2.0f;
    float bar_h = btn_h + pad_bar_y * 2.0f;
    float bar_x = ((float)fb_w - bar_w) * 0.5f;
    float bar_y = (float)fb_h - 20.0f * dpi - bar_h;

    /* Check hover & auto-hide */
    int is_mouse_over_bar = (mouse_fb_x >= bar_x && mouse_fb_x <= bar_x + bar_w &&
                             mouse_fb_y >= bar_y && mouse_fb_y <= bar_y + bar_h);
    if (is_mouse_over_bar)
        r->last_mouse_move_ns = now_ns;

    float idle_sec = (float)(now_ns - r->last_mouse_move_ns) / 1e9f;
    float alpha = 1.0f;
    if (idle_sec > 3.0f) {
        alpha = 1.0f - (idle_sec - 3.0f) / 0.5f;
        if (alpha < 0.0f) alpha = 0.0f;
    }
    if (alpha <= 0.01f)
        return; /* Fully hidden during inactive video watching */

    /* 1. Toolbar floating container */
    add_solid_rect(r, bar_x, bar_y, bar_w, bar_h, 0.04f, 0.06f, 0.09f, alpha * 0.85f);
    float border_t = 1.0f * dpi;
    add_solid_rect(r, bar_x, bar_y, bar_w, border_t, 0.22f, 0.27f, 0.35f, alpha * 0.70f);
    add_solid_rect(r, bar_x, bar_y + bar_h - border_t, bar_w, border_t, 0.22f, 0.27f, 0.35f, alpha * 0.70f);
    add_solid_rect(r, bar_x, bar_y, border_t, bar_h, 0.22f, 0.27f, 0.35f, alpha * 0.70f);
    add_solid_rect(r, bar_x + bar_w - border_t, bar_y, border_t, bar_h, 0.22f, 0.27f, 0.35f, alpha * 0.70f);

    /* 2. Render buttons */
    float cur_x = bar_x + pad_bar_x;
    float btn_y = bar_y + pad_bar_y;

    for (int i = 0; i < num_buttons; i++) {
        float bw = btn_widths[i];
        int hovered = (mouse_fb_x >= cur_x && mouse_fb_x <= cur_x + bw &&
                       mouse_fb_y >= btn_y && mouse_fb_y <= btn_y + btn_h);

        if (hovered && r->mouse_clicked) {
            trigger_action(r, buttons[i].action);
            r->mouse_clicked = 0;
        }

        /* Button background & border colors */
        float bg_r, bg_g, bg_b, bg_a;
        float bdr_r, bdr_g, bdr_b, bdr_a;
        float txt_r, txt_g, txt_b;

        if (hovered) {
            bg_r = 0.20f; bg_g = 0.26f; bg_b = 0.38f; bg_a = alpha * 0.95f;
            bdr_r = 0.45f; bdr_g = 0.75f; bdr_b = 1.00f; bdr_a = alpha * 0.95f;
            txt_r = 1.00f; txt_g = 1.00f; txt_b = 1.00f;
        } else if (buttons[i].active) {
            bg_r = 0.10f; bg_g = 0.25f; bg_b = 0.20f; bg_a = alpha * 0.85f;
            bdr_r = 0.25f; bdr_g = 0.80f; bdr_b = 0.55f; bdr_a = alpha * 0.85f;
            txt_r = 0.45f; txt_g = 0.95f; txt_b = 0.70f;
        } else {
            bg_r = 0.08f; bg_g = 0.11f; bg_b = 0.16f; bg_a = alpha * 0.80f;
            bdr_r = 0.25f; bdr_g = 0.30f; bdr_b = 0.38f; bdr_a = alpha * 0.65f;
            txt_r = 0.85f; txt_g = 0.88f; txt_b = 0.92f;
        }

        add_solid_rect(r, cur_x, btn_y, bw, btn_h, bg_r, bg_g, bg_b, bg_a);
        add_solid_rect(r, cur_x, btn_y, bw, border_t, bdr_r, bdr_g, bdr_b, bdr_a);
        add_solid_rect(r, cur_x, btn_y + btn_h - border_t, bw, border_t, bdr_r, bdr_g, bdr_b, bdr_a);
        add_solid_rect(r, cur_x, btn_y, border_t, btn_h, bdr_r, bdr_g, bdr_b, bdr_a);
        add_solid_rect(r, cur_x + bw - border_t, btn_y, border_t, btn_h, bdr_r, bdr_g, bdr_b, bdr_a);

        add_string(r, buttons[i].label, cur_x + pad_btn_x, btn_y + pad_btn_y,
                   char_w, char_h, txt_r, txt_g, txt_b, alpha);

        cur_x += bw + spacing;
    }

    /* 3. Render floating toast notification */
    if (now_ns < r->toast_expire_ns && r->toast_text[0]) {
        uint64_t rem = r->toast_expire_ns - now_ns;
        float toast_alpha = (rem < 400000000ULL) ? ((float)rem / 400000000.0f) : 1.0f;

        float toast_w = (float)strlen(r->toast_text) * char_w + pad_btn_x * 2.0f;
        float toast_h = char_h + pad_btn_y * 1.5f;
        float toast_x = ((float)fb_w - toast_w) * 0.5f;
        if (toast_x < 10.0f * dpi) toast_x = 10.0f * dpi;
        float toast_y = bar_y - toast_h - 12.0f * dpi;

        add_solid_rect(r, toast_x, toast_y, toast_w, toast_h, 0.05f, 0.07f, 0.11f, toast_alpha * 0.92f);
        add_solid_rect(r, toast_x, toast_y, toast_w, border_t, 0.25f, 0.65f, 0.95f, toast_alpha * 0.85f);
        add_solid_rect(r, toast_x, toast_y + toast_h - border_t, toast_w, border_t, 0.25f, 0.65f, 0.95f, toast_alpha * 0.85f);
        add_solid_rect(r, toast_x, toast_y, border_t, toast_h, 0.25f, 0.65f, 0.95f, toast_alpha * 0.85f);
        add_solid_rect(r, toast_x + toast_w - border_t, toast_y, border_t, toast_h, 0.25f, 0.65f, 0.95f, toast_alpha * 0.85f);

        float text_y = toast_y + (toast_h - char_h) * 0.5f;
        add_string(r, r->toast_text, toast_x + pad_btn_x, text_y,
                   char_w, char_h, 1.0f, 1.0f, 1.0f, toast_alpha);
    }
}

void render_frame(renderer_t *r, const frame_t *f)
{
    int fb_w = 0, fb_h = 0, win_w = 0, win_h = 0;
    SDL_GetWindowSizeInPixels(r->win, &fb_w, &fb_h);
    SDL_GetWindowSize(r->win, &win_w, &win_h);
    if (fb_w <= 0 || fb_h <= 0)
        return;

    glViewport(0, 0, fb_w, fb_h);

    /* Decay shutter flash animation */
    if (r->flash > 0.01f)
        r->flash *= 0.78f;
    else
        r->flash = 0.0f;

    /* Upload video frame (skip upload when paused to hold the frozen frame) */
    if (!r->paused) {
        if (f->fmt == PIXFMT_RGB24)
            upload_rgb(r, f);
        else /* NV12 */
            upload_nv12(r, f);
    }

    /* Render video quad */
    glBindVertexArray(r->vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    /* Process clean snapshot capture before drawing HUD/toolbar */
    if (r->snapshot_requested) {
        r->snapshot_requested = 0;
        take_snapshot(r, fb_w, fb_h);
    }

    /* Build overlay batch: stats HUD + bottom toolbar */
    r->nverts = 0;
    render_stats_overlay(r, f, fb_w, fb_h, win_w, win_h);
    render_gui_toolbar(r, fb_w, fb_h, win_w);

    if (r->nverts > 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(r->prog_overlay);
        glUniform2f(glGetUniformLocation(r->prog_overlay, "u_screen"), (float)fb_w, (float)fb_h);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, r->tex_font);
        glUniform1i(glGetUniformLocation(r->prog_overlay, "u_font"), 0);

        glBindVertexArray(r->overlay_vao);
        glBindBuffer(GL_ARRAY_BUFFER, r->overlay_vbo);
        glBufferData(GL_ARRAY_BUFFER, (ptrdiff_t)(r->nverts * 8 * sizeof(float)), r->verts, GL_DYNAMIC_DRAW);

        glDrawArrays(GL_TRIANGLES, 0, r->nverts);

        glDisable(GL_BLEND);
    }

    r->mouse_clicked = 0; /* Reset single click */
    SDL_GL_SwapWindow(r->win);
}

int render_poll_quit(void)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) {
            return 1;
        } else if (e.type == SDL_EVENT_MOUSE_MOTION) {
            if (g_current_renderer) {
                g_current_renderer->mouse_win_x = e.motion.x;
                g_current_renderer->mouse_win_y = e.motion.y;
                g_current_renderer->last_mouse_move_ns = SDL_GetTicksNS();
            }
        } else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
            if (e.button.button == SDL_BUTTON_LEFT && g_current_renderer) {
                g_current_renderer->mouse_clicked = 1;
                g_current_renderer->mouse_win_x = e.button.x;
                g_current_renderer->mouse_win_y = e.button.y;
                g_current_renderer->last_mouse_move_ns = SDL_GetTicksNS();
            }
        } else if (e.type == SDL_EVENT_KEY_DOWN) {
            if (e.key.key == SDLK_ESCAPE) {
                return 1;
            } else if (e.key.key == SDLK_M) {
                if (g_current_renderer)
                    trigger_action(g_current_renderer, ACTION_MIRROR);
            } else if (e.key.key == SDLK_SPACE || e.key.key == SDLK_P) {
                if (g_current_renderer)
                    trigger_action(g_current_renderer, ACTION_PAUSE);
            } else if (e.key.key == SDLK_C || e.key.key == SDLK_RETURN) {
                if (g_current_renderer)
                    trigger_action(g_current_renderer, ACTION_SNAPSHOT);
            } else if (e.key.key == SDLK_TAB) {
                if (g_current_renderer)
                    trigger_action(g_current_renderer, ACTION_FILTER);
            } else if (e.key.key == SDLK_S || e.key.key == SDLK_F3) {
                if (g_current_renderer)
                    trigger_action(g_current_renderer, ACTION_STATS);
            } else if (e.key.key == SDLK_F) {
                if (g_current_renderer && g_current_renderer->win) {
                    bool fs = (SDL_GetWindowFlags(g_current_renderer->win) & SDL_WINDOW_FULLSCREEN) != 0;
                    SDL_SetWindowFullscreen(g_current_renderer->win, !fs);
                }
            }
        }
    }
    return 0;
}

void render_destroy(renderer_t *r)
{
    if (!r)
        return;
    if (g_current_renderer == r)
        g_current_renderer = NULL;
    glDeleteProgram(r->prog_nv12);
    glDeleteProgram(r->prog_rgb);
    glDeleteProgram(r->prog_overlay);
    glDeleteTextures(1, &r->tex_y);
    glDeleteTextures(1, &r->tex_uv);
    glDeleteTextures(1, &r->tex_font);
    SDL_GL_DestroyContext(r->gl);
    SDL_DestroyWindow(r->win);
    SDL_Quit();
    free(r);
}
