/*
 * qchem_shim.c — Fast C replacement for the Python nonmpi_qc_shim.
 *
 * Eliminates ~50-100 ms Python startup overhead per MD step.
 *
 * Usage (called by AMBER):
 *     qchem <inpfile> <logfile> <savfile>
 *
 * Environment variables:
 *     AMBER_MLIPS_SERVER_SOCKET  path to Unix domain socket
 *     AMBER_MLIPS_BACKEND        backend name (informational)
 *     AMBER_MLIPS_ML_KEYWORDS    ml_keywords string (checked for --embedcharge)
 *
 * Wire protocol: 4-byte big-endian length prefix + payload.
 *   Payload format auto-detected by first byte:
 *     '{' (0x7B) → legacy JSON
 *     0x01      → binary: [0x01][4B json_len][json_meta][array data...]
 *   Binary format avoids sprintf/sscanf overhead for large numeric arrays.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>   /* htonl / ntohl */

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */
#define HARTREE_PER_EV   (1.0 / 27.211386245988)
#define BOHR_PER_ANG     1.8897261254578281
/* force (eV/Å) → gradient (Eh/Bohr) = -HARTREE_PER_EV / BOHR_PER_ANG */
#define GRAD_FACTOR      (-(HARTREE_PER_EV) / (BOHR_PER_ANG))

#define MAX_ATOMS   8192
#define MAX_MM     65536
#define MAX_JSON   (64 * 1024 * 1024)   /* 64 MB */
#define LINE_BUF    4096

/* ------------------------------------------------------------------ */
/* QM atom                                                            */
/* ------------------------------------------------------------------ */
typedef struct { char sym[4]; double x, y, z; } QMAtom;

/* ------------------------------------------------------------------ */
/* Minimal JSON helpers (no external dependency)                      */
/* ------------------------------------------------------------------ */

/* Skip whitespace. */
static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Find a key in a JSON object (flat, non-recursive). Returns pointer
   past the colon, or NULL. */
static const char *json_find_key(const char *json, const char *key) {
    char needle[256];
    int n = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (n < 0 || n >= (int)sizeof(needle)) return NULL;
    const char *p = strstr(json, needle);
    if (!p) return NULL;
    p += n;
    p = skip_ws(p);
    if (*p != ':') return NULL;
    p++;
    return skip_ws(p);
}

/* Parse a JSON number at *p. Advances *p past the number. */
static double json_parse_number(const char **pp) {
    char buf[64];
    const char *start = *pp;
    const char *p = start;
    if (*p == '-' || *p == '+') p++;
    while ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' ||
           *p == 'E' || *p == '+' || *p == '-') p++;
    int len = (int)(p - start);
    if (len <= 0 || len >= (int)sizeof(buf)) { *pp = p; return 0.0; }
    memcpy(buf, start, len);
    buf[len] = '\0';
    *pp = p;
    return strtod(buf, NULL);
}

/* Parse a JSON string value at *p (expects opening "). Returns pointer
   to static buffer. Advances *p past closing ". */
static const char *json_parse_string(const char **pp) {
    static char sbuf[256];
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < (int)sizeof(sbuf) - 1) {
        sbuf[i++] = *p++;
    }
    sbuf[i] = '\0';
    if (*p == '"') p++;
    *pp = p;
    return sbuf;
}

/* ------------------------------------------------------------------ */
/* Parse Q-Chem input                                                 */
/* ------------------------------------------------------------------ */
static int parse_qchem_input(
    const char *inpfile,
    int *charge, int *spinmult,
    QMAtom *qm, int *nqm,
    double mm_coords[][3], double *mm_charges, int *nmm
) {
    FILE *fp = fopen(inpfile, "r");
    if (!fp) { fprintf(stderr, "[c-shim] Cannot open %s: %s\n", inpfile, strerror(errno)); return -1; }

    char line[LINE_BUF];
    int in_molecule = 0, in_extchg = 0, mol_header_read = 0;
    *nqm = 0; *nmm = 0; *charge = 0; *spinmult = 1;

    while (fgets(line, sizeof(line), fp)) {
        /* trim leading whitespace */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        /* trim trailing newline */
        char *nl = strchr(p, '\n');
        if (nl) *nl = '\0';
        nl = strchr(p, '\r');
        if (nl) *nl = '\0';

        if (p[0] == '\0') continue;

        /* Section markers */
        if (p[0] == '$') {
            char sec[64] = {0};
            int si = 0;
            char *sp = p + 1;
            while (*sp && *sp != ' ' && *sp != '\t' && si < 62) {
                sec[si++] = (*sp >= 'A' && *sp <= 'Z') ? (*sp + 32) : *sp;
                sp++;
            }
            sec[si] = '\0';

            if (strcmp(sec, "end") == 0) {
                in_molecule = 0;
                in_extchg = 0;
            } else if (strcmp(sec, "molecule") == 0) {
                in_molecule = 1; mol_header_read = 0;
            } else if (strcmp(sec, "external_charges") == 0) {
                in_extchg = 1;
            } else {
                in_molecule = 0;
                in_extchg = 0;
            }
            continue;
        }

        if (in_molecule) {
            if (!mol_header_read) {
                /* First line: charge multiplicity */
                if (sscanf(p, "%d %d", charge, spinmult) < 2) {
                    fprintf(stderr, "[c-shim] Failed to parse charge/mult from: %s\n", p);
                    fclose(fp); return -1;
                }
                mol_header_read = 1;
                continue;
            }
            /* Atom line: sym x y z */
            if (*nqm >= MAX_ATOMS) { fprintf(stderr, "[c-shim] Too many QM atoms\n"); fclose(fp); return -1; }
            char sym[8];
            double x, y, z;
            if (sscanf(p, "%7s %lf %lf %lf", sym, &x, &y, &z) == 4) {
                /* Normalise symbol */
                QMAtom *a = &qm[*nqm];
                a->sym[0] = (sym[0] >= 'a' && sym[0] <= 'z') ? (sym[0] - 32) : sym[0];
                if (sym[1] && sym[1] != ' ') {
                    a->sym[1] = (sym[1] >= 'A' && sym[1] <= 'Z') ? (sym[1] + 32) : sym[1];
                    a->sym[2] = '\0';
                } else {
                    a->sym[1] = '\0';
                }
                a->x = x; a->y = y; a->z = z;
                (*nqm)++;
            }
            continue;
        }

        if (in_extchg) {
            if (*nmm >= MAX_MM) { fprintf(stderr, "[c-shim] Too many MM atoms\n"); fclose(fp); return -1; }
            double x, y, z, q;
            if (sscanf(p, "%lf %lf %lf %lf", &x, &y, &z, &q) == 4) {
                mm_coords[*nmm][0] = x;
                mm_coords[*nmm][1] = y;
                mm_coords[*nmm][2] = z;
                mm_charges[*nmm] = q;
                (*nmm)++;
            }
            continue;
        }
    }
    fclose(fp);

    if (*nqm == 0) { fprintf(stderr, "[c-shim] No QM atoms found in %s\n", inpfile); return -1; }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Build JSON request                                                 */
/* ------------------------------------------------------------------ */
/* Legacy JSON builder — kept for reference / fallback testing. */
static char *build_request_json(
    const QMAtom *, int, int, int,
    const double [][3], const double *, int
) __attribute__((unused));
static char *build_request_json(
    const QMAtom *qm, int nqm,
    int charge, int spinmult,
    const double mm_coords[][3], const double *mm_charges, int nmm
) {
    /* Estimate size: ~60 bytes per QM atom + ~80 per MM atom + overhead */
    size_t cap = (size_t)(nqm * 60 + nmm * 80 + 512);
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    int off = 0;
    off += snprintf(buf + off, cap - off,
        "{\"action\":\"evaluate\","
        "\"symbols\":[");

    for (int i = 0; i < nqm; i++) {
        off += snprintf(buf + off, cap - off,
            "%s\"%s\"", i ? "," : "", qm[i].sym);
    }
    off += snprintf(buf + off, cap - off, "],\"coords_ang\":[");

    for (int i = 0; i < nqm; i++) {
        off += snprintf(buf + off, cap - off,
            "%s[%.16g,%.16g,%.16g]",
            i ? "," : "", qm[i].x, qm[i].y, qm[i].z);
    }
    off += snprintf(buf + off, cap - off,
        "],\"charge\":%d,\"multiplicity\":%d,"
        "\"need_forces\":true,\"need_hessian\":false,"
        "\"hessian_mode\":\"analytical\",\"hessian_step\":0.001",
        charge, spinmult);

    /* Append MM coordinates and charges so server can do embedcharge. */
    if (nmm > 0) {
        /* Grow buffer if needed */
        size_t need = (size_t)(off + nmm * 80 + 128);
        if (need > cap) {
            cap = need;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        off += snprintf(buf + off, cap - off, ",\"mm_coords_ang\":[");
        for (int i = 0; i < nmm; i++) {
            off += snprintf(buf + off, cap - off,
                "%s[%.16g,%.16g,%.16g]",
                i ? "," : "", mm_coords[i][0], mm_coords[i][1], mm_coords[i][2]);
            /* Grow if approaching limit */
            if ((size_t)off + 128 > cap) {
                cap *= 2;
                char *nb = (char *)realloc(buf, cap);
                if (!nb) { free(buf); return NULL; }
                buf = nb;
            }
        }
        off += snprintf(buf + off, cap - off, "],\"mm_charges\":[");
        for (int i = 0; i < nmm; i++) {
            off += snprintf(buf + off, cap - off,
                "%s%.16g", i ? "," : "", mm_charges[i]);
            if ((size_t)off + 64 > cap) {
                cap *= 2;
                char *nb = (char *)realloc(buf, cap);
                if (!nb) { free(buf); return NULL; }
                buf = nb;
            }
        }
        off += snprintf(buf + off, cap - off, "]");
    }

    off += snprintf(buf + off, cap - off, "}");
    return buf;
}

/* ------------------------------------------------------------------ */
/* Socket I/O (length-prefixed JSON)                                  */
/* ------------------------------------------------------------------ */
static int send_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) { if (errno == EINTR) continue; return -1; }
        p += n; len -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n <= 0) { if (n == 0) return -1; if (errno == EINTR) continue; return -1; }
        p += n; len -= (size_t)n;
    }
    return 0;
}

/* Legacy JSON send — used for ping/shutdown; suppressed unused warning. */
static int send_msg(int fd, const char *json_str) __attribute__((unused));
static int send_msg(int fd, const char *json_str) {
    uint32_t len = (uint32_t)strlen(json_str);
    uint32_t net_len = htonl(len);
    if (send_all(fd, &net_len, 4) != 0) return -1;
    if (send_all(fd, json_str, len) != 0) return -1;
    return 0;
}

static char *recv_msg(int fd) __attribute__((unused));
static char *recv_msg(int fd) {
    uint32_t net_len;
    if (recv_all(fd, &net_len, 4) != 0) return NULL;
    uint32_t len = ntohl(net_len);
    if (len > MAX_JSON) { fprintf(stderr, "[c-shim] Response too large: %u\n", len); return NULL; }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) return NULL;
    if (recv_all(fd, buf, len) != 0) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

/* ------------------------------------------------------------------ */
/* Binary wire protocol helpers                                       */
/*                                                                    */
/* Binary payload: [0x01][4B json_len][json_meta][array data...]      */
/* Arrays described in json_meta["_bin"]: {"name": [dim0, dim1, ...]} */
/* Array data is raw float64 in declaration order, native byte order. */
/* ------------------------------------------------------------------ */
#define BIN_MAGIC 0x01

/* Build and send a binary-format request.  Returns 0 on success. */
static int send_request_bin(
    int fd,
    const QMAtom *qm, int nqm,
    int charge, int spinmult,
    const double mm_coords[][3], const double *mm_charges, int nmm
) {
    /* 1) Build JSON metadata (small: symbols + scalars + _bin descriptor) */
    size_t cap = (size_t)(nqm * 10 + 512);
    char *meta = (char *)malloc(cap);
    if (!meta) return -1;

    int off = 0;
    off += snprintf(meta + off, cap - off,
        "{\"action\":\"evaluate\","
        "\"symbols\":[");
    for (int i = 0; i < nqm; i++) {
        off += snprintf(meta + off, cap - off,
            "%s\"%s\"", i ? "," : "", qm[i].sym);
    }
    off += snprintf(meta + off, cap - off,
        "],\"charge\":%d,\"multiplicity\":%d,"
        "\"need_forces\":true,\"need_hessian\":false,"
        "\"hessian_mode\":\"analytical\",\"hessian_step\":0.001,"
        "\"_bin\":{\"coords_ang\":[%d,3]",
        charge, spinmult, nqm);
    if (nmm > 0) {
        off += snprintf(meta + off, cap - off,
            ",\"mm_coords_ang\":[%d,3],\"mm_charges\":[%d]",
            nmm, nmm);
    }
    off += snprintf(meta + off, cap - off, "}}");

    uint32_t json_len = (uint32_t)off;

    /* 2) Compute array data sizes */
    size_t coords_bytes = (size_t)nqm * 3 * sizeof(double);
    size_t mm_coords_bytes = (nmm > 0) ? (size_t)nmm * 3 * sizeof(double) : 0;
    size_t mm_charges_bytes = (nmm > 0) ? (size_t)nmm * sizeof(double) : 0;
    size_t array_total = coords_bytes + mm_coords_bytes + mm_charges_bytes;

    /* 3) Assemble binary payload: magic(1) + json_len(4) + json + arrays */
    size_t payload_len = 1 + 4 + json_len + array_total;
    char *payload = (char *)malloc(payload_len);
    if (!payload) { free(meta); return -1; }

    size_t p = 0;
    payload[p++] = (char)BIN_MAGIC;

    uint32_t net_jlen = htonl(json_len);
    memcpy(payload + p, &net_jlen, 4); p += 4;
    memcpy(payload + p, meta, json_len); p += json_len;
    free(meta);

    /* Pack coords_ang as contiguous float64 */
    for (int i = 0; i < nqm; i++) {
        double v[3] = { qm[i].x, qm[i].y, qm[i].z };
        memcpy(payload + p, v, 3 * sizeof(double)); p += 3 * sizeof(double);
    }
    /* Pack mm_coords_ang */
    if (nmm > 0 && mm_coords_bytes > 0) {
        memcpy(payload + p, mm_coords, mm_coords_bytes); p += mm_coords_bytes;
    }
    /* Pack mm_charges */
    if (nmm > 0 && mm_charges_bytes > 0) {
        memcpy(payload + p, mm_charges, mm_charges_bytes); p += mm_charges_bytes;
    }

    /* 4) Send: [4-byte total length][payload] */
    uint32_t net_total = htonl((uint32_t)payload_len);
    int rc = send_all(fd, &net_total, 4);
    if (rc == 0) rc = send_all(fd, payload, payload_len);
    free(payload);
    return rc;
}

/* Receive a response, auto-detecting JSON or binary format.
 * Extracts energy, QM forces, and optional MM forces.
 * Returns 0 on success, -1 on error. */
static int recv_and_parse_response(
    int fd, int nqm, int nmm,
    double *energy_ev,
    double forces_ev_ang[][3],
    double forces_mm_ev_ang[][3]
) {
    uint32_t net_len;
    if (recv_all(fd, &net_len, 4) != 0) return -1;
    uint32_t len = ntohl(net_len);
    if (len > MAX_JSON) { fprintf(stderr, "[c-shim] Response too large: %u\n", len); return -1; }

    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) return -1;
    if (recv_all(fd, buf, len) != 0) { free(buf); return -1; }
    buf[len] = '\0';

    if (len > 0 && (unsigned char)buf[0] == BIN_MAGIC) {
        /* Binary format response */
        if (len < 5) { free(buf); return -1; }
        uint32_t net_jlen;
        memcpy(&net_jlen, buf + 1, 4);
        uint32_t jlen = ntohl(net_jlen);
        char *json_meta = buf + 5;
        json_meta[jlen] = '\0';  /* null-terminate JSON portion */

        /* Parse status */
        const char *st = json_find_key(json_meta, "status");
        if (!st || *st != '"') {
            fprintf(stderr, "[c-shim] No status in binary response\n");
            free(buf); return -1;
        }
        const char *status_str = json_parse_string(&st);
        if (strcmp(status_str, "ok") != 0) {
            const char *msg = json_find_key(json_meta, "message");
            if (msg && *msg == '"') {
                const char *m = json_parse_string(&msg);
                fprintf(stderr, "[c-shim] Server error: %s\n", m);
            }
            free(buf); return -1;
        }

        /* Parse energy_ev from JSON meta */
        const char *ep = json_find_key(json_meta, "energy_ev");
        if (!ep) { fprintf(stderr, "[c-shim] No energy_ev\n"); free(buf); return -1; }
        *energy_ev = json_parse_number(&ep);

        /* Read binary arrays from after JSON portion.
         * Order matches _bin descriptor: forces_ev_ang, then forces_mm_ev_ang (if present). */
        size_t data_off = 5 + jlen;
        size_t f_bytes = (size_t)nqm * 3 * sizeof(double);
        if (data_off + f_bytes <= len) {
            memcpy(forces_ev_ang, buf + data_off, f_bytes);
            data_off += f_bytes;
        } else {
            /* No binary force data → zero */
            memset(forces_ev_ang, 0, f_bytes);
        }

        if (nmm > 0) {
            size_t fm_bytes = (size_t)nmm * 3 * sizeof(double);
            if (data_off + fm_bytes <= len) {
                memcpy(forces_mm_ev_ang, buf + data_off, fm_bytes);
                data_off += fm_bytes;
            } else {
                memset(forces_mm_ev_ang, 0, fm_bytes);
            }
        }

        free(buf);
        return 0;

    } else {
        /* Legacy JSON format response */
        const char *st = json_find_key(buf, "status");
        if (!st || *st != '"') {
            fprintf(stderr, "[c-shim] No status in response\n");
            free(buf); return -1;
        }
        const char *status_str = json_parse_string(&st);
        if (strcmp(status_str, "ok") != 0) {
            const char *msg = json_find_key(buf, "message");
            if (msg && *msg == '"') {
                const char *m = json_parse_string(&msg);
                fprintf(stderr, "[c-shim] Server error: %s\n", m);
            } else {
                fprintf(stderr, "[c-shim] Server returned status: %s\n", status_str);
            }
            free(buf); return -1;
        }

        /* Parse energy_ev */
        const char *ep2 = json_find_key(buf, "energy_ev");
        if (!ep2) { fprintf(stderr, "[c-shim] No energy_ev in response\n"); free(buf); return -1; }
        *energy_ev = json_parse_number(&ep2);

        /* Parse forces_ev_ang */
        const char *fp2 = json_find_key(buf, "forces_ev_ang");
        if (!fp2 || *fp2 == 'n') {
            for (int i = 0; i < nqm; i++) { forces_ev_ang[i][0] = forces_ev_ang[i][1] = forces_ev_ang[i][2] = 0.0; }
        } else {
            fp2 = skip_ws(fp2);
            if (*fp2 != '[') { fprintf(stderr, "[c-shim] Expected '[' for forces\n"); free(buf); return -1; }
            fp2++;
            for (int i = 0; i < nqm; i++) {
                fp2 = skip_ws(fp2); if (*fp2 == ',') fp2++; fp2 = skip_ws(fp2);
                if (*fp2 != '[') { fprintf(stderr, "[c-shim] Expected '[' for force[%d]\n", i); free(buf); return -1; }
                fp2++;
                forces_ev_ang[i][0] = json_parse_number(&fp2);
                fp2 = skip_ws(fp2); if (*fp2 == ',') fp2++; fp2 = skip_ws(fp2);
                forces_ev_ang[i][1] = json_parse_number(&fp2);
                fp2 = skip_ws(fp2); if (*fp2 == ',') fp2++; fp2 = skip_ws(fp2);
                forces_ev_ang[i][2] = json_parse_number(&fp2);
                fp2 = skip_ws(fp2); if (*fp2 == ']') fp2++;
            }
        }

        /* Parse forces_mm_ev_ang */
        if (nmm > 0) {
            memset(forces_mm_ev_ang, 0, sizeof(double) * 3 * nmm);
            const char *mp2 = json_find_key(buf, "forces_mm_ev_ang");
            if (mp2 && *mp2 == '[') {
                mp2++;
                for (int i = 0; i < nmm; i++) {
                    mp2 = skip_ws(mp2); if (*mp2 == ',') mp2++; mp2 = skip_ws(mp2);
                    if (*mp2 != '[') break;
                    mp2++;
                    forces_mm_ev_ang[i][0] = json_parse_number(&mp2);
                    mp2 = skip_ws(mp2); if (*mp2 == ',') mp2++; mp2 = skip_ws(mp2);
                    forces_mm_ev_ang[i][1] = json_parse_number(&mp2);
                    mp2 = skip_ws(mp2); if (*mp2 == ',') mp2++; mp2 = skip_ws(mp2);
                    forces_mm_ev_ang[i][2] = json_parse_number(&mp2);
                    mp2 = skip_ws(mp2); if (*mp2 == ']') mp2++;
                }
            }
        }

        free(buf);
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Connect to server and evaluate                                     */
/* ------------------------------------------------------------------ */
static int evaluate_via_server(
    const char *socket_path,
    const QMAtom *qm, int nqm,
    int charge, int spinmult,
    const double mm_coords[][3], const double *mm_charges, int nmm,
    double *energy_ev,               /* out */
    double forces_ev_ang[][3],       /* out: nqm x 3 */
    double forces_mm_ev_ang[][3],    /* out: nmm x 3, may be zero */
    int *ok                          /* out: 1=success */
) {
    *ok = 0;

    /* Connect */
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { fprintf(stderr, "[c-shim] socket: %s\n", strerror(errno)); return -1; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "[c-shim] connect(%s): %s\n", socket_path, strerror(errno));
        close(fd); return -1;
    }

    /* Send binary request */
    if (send_request_bin(fd, qm, nqm, charge, spinmult, mm_coords, mm_charges, nmm) != 0) {
        fprintf(stderr, "[c-shim] send failed\n");
        close(fd); return -1;
    }

    /* Receive and parse response (auto-detects JSON or binary) */
    int rc = recv_and_parse_response(fd, nqm, nmm, energy_ev, forces_ev_ang, forces_mm_ev_ang);
    close(fd);

    if (rc != 0) { fprintf(stderr, "[c-shim] recv/parse failed\n"); return -1; }

    *ok = 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Write output files                                                 */
/* ------------------------------------------------------------------ */
static int write_outputs(
    const char *logfile, const char *savfile,
    double energy_ha,
    const double efield_mm[][3], int nmm,
    const double grad_qm[][3], int nqm
) {
    /* logfile */
    FILE *fp = fopen(logfile, "w");
    if (!fp) { fprintf(stderr, "[c-shim] Cannot write %s: %s\n", logfile, strerror(errno)); return -1; }
    fprintf(fp, " AMBER-MLIPS qchem shim\n");
    fprintf(fp, " Charge-charge energy = %17.10f\n", 0.0);
    fprintf(fp, " SCF   energy in the final basis set = % .16f\n", energy_ha);
    fclose(fp);

    /* efield.dat */
    fp = fopen("efield.dat", "w");
    if (!fp) { fprintf(stderr, "[c-shim] Cannot write efield.dat: %s\n", strerror(errno)); return -1; }
    for (int i = 0; i < nmm; i++) {
        fprintf(fp, "%22.16f%22.16f%22.16f\n", efield_mm[i][0], efield_mm[i][1], efield_mm[i][2]);
    }
    for (int i = 0; i < nqm; i++) {
        fprintf(fp, "%25.20f%25.20f%25.20f\n", grad_qm[i][0], grad_qm[i][1], grad_qm[i][2]);
    }
    fclose(fp);

    /* savfile */
    fp = fopen(savfile, "w");
    if (!fp) { fprintf(stderr, "[c-shim] Cannot write %s: %s\n", savfile, strerror(errno)); return -1; }
    fprintf(fp, "# amber-mlips qchem shim checkpoint marker\n");
    fclose(fp);

    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: qchem <inpfile> <logfile> <savfile>\n");
        return 2;
    }

    const char *inpfile = argv[1];
    const char *logfile = argv[2];
    const char *savfile = argv[3];

    const char *socket_path = getenv("AMBER_MLIPS_SERVER_SOCKET");
    if (!socket_path || !socket_path[0]) {
        fprintf(stderr, "[c-shim] ERROR: AMBER_MLIPS_SERVER_SOCKET not set\n");
        return 1;
    }

    /* Parse Q-Chem input */
    static QMAtom qm[MAX_ATOMS];
    static double mm_coords[MAX_MM][3];
    static double mm_charges[MAX_MM];
    int nqm = 0, nmm = 0, charge = 0, spinmult = 1;

    if (parse_qchem_input(inpfile, &charge, &spinmult, qm, &nqm, mm_coords, mm_charges, &nmm) != 0)
        return 1;

    /* Evaluate via server (MM data included for server-side embedcharge) */
    double energy_ev = 0.0;
    static double forces_ev_ang[MAX_ATOMS][3];
    static double forces_mm_ev_ang[MAX_MM][3];
    int ok = 0;

    if (evaluate_via_server(socket_path, qm, nqm, charge, spinmult,
                            mm_coords, mm_charges, nmm,
                            &energy_ev, forces_ev_ang, forces_mm_ev_ang, &ok) != 0 || !ok)
        return 1;

    /* Convert units: eV → Eh, eV/Å → Eh/Bohr (gradient = -force * factor) */
    double energy_ha = energy_ev * HARTREE_PER_EV;

    static double grad_qm[MAX_ATOMS][3];
    for (int i = 0; i < nqm; i++) {
        grad_qm[i][0] = forces_ev_ang[i][0] * GRAD_FACTOR;
        grad_qm[i][1] = forces_ev_ang[i][1] * GRAD_FACTOR;
        grad_qm[i][2] = forces_ev_ang[i][2] * GRAD_FACTOR;
    }

    /* MM efield: E = -grad / q (AMBER converts back: grad = -E * q) */
    static double efield_mm[MAX_MM][3];
    for (int i = 0; i < nmm; i++) {
        double grad_x = forces_mm_ev_ang[i][0] * GRAD_FACTOR;
        double grad_y = forces_mm_ev_ang[i][1] * GRAD_FACTOR;
        double grad_z = forces_mm_ev_ang[i][2] * GRAD_FACTOR;
        double q = mm_charges[i];
        if (fabs(q) > 1.0e-14) {
            efield_mm[i][0] = -grad_x / q;
            efield_mm[i][1] = -grad_y / q;
            efield_mm[i][2] = -grad_z / q;
        } else {
            efield_mm[i][0] = efield_mm[i][1] = efield_mm[i][2] = 0.0;
        }
    }

    /* Write outputs */
    if (write_outputs(logfile, savfile, energy_ha, efield_mm, nmm, grad_qm, nqm) != 0)
        return 1;

    return 0;
}
