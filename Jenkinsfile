pipeline {
    agent { label 'linux-astree' } // Assumes a Linux agent with this label is configured in Jenkins.

    options {
        timestamps()
        disableConcurrentBuilds() // Additional safety; lock is also used during analysis.
    }

    parameters {
        string(name: 'ANALYSIS_NAME', defaultValue: '', description: 'Analysis name')
        string(name: 'PROJECT_ID', defaultValue: '', description: 'Project identifier')
        string(name: 'ASTREE_VERSION', defaultValue: '', description: 'Astrée version')
        string(name: 'CONFIG_PROFILE', defaultValue: '', description: 'Config profile')
        string(name: 'REPO_URL', defaultValue: '', description: 'Git repository URL')
        string(name: 'BRANCH', defaultValue: 'main', description: 'Git branch')
        string(name: 'KLOC', defaultValue: '', description: 'KLOC (optional)')
        string(name: 'MAX_MEM_GB', defaultValue: '32', description: 'Max memory (GB)')
        choice(name: 'TIMEOUT_HOURS', choices: ['2', '6', '12', '24', '48', '72'], description: 'Analysis timeout in hours')
        string(name: 'EMAIL_TO', defaultValue: '', description: 'Email recipients')
        booleanParam(name: 'DRY_RUN', defaultValue: false, description: 'Skip actual analysis execution')
    }

    environment {
        REPO_CREDENTIALS_ID = 'git-credentials-id' // Placeholder credential ID for private repos.
        LOCK_RESOURCE = 'astree-linux'
        HEARTBEAT_MIN = '5'
        SAFE_MIN_FREE_GB = '2'
        SUMMARY_JSON = 'analysis_summary.json'
        LAST_MEM_JSON = 'last_mem_stats.json'
    }

    stages {
        stage('Init') {
            steps {
                script {
                    def queuedMillis = currentBuild.rawBuild.getTimeInMillis()
                    def startMillis = System.currentTimeMillis()
                    env.QUEUED_TIME_UTC = new Date(queuedMillis).format("yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone.getTimeZone('UTC'))
                    env.BUILD_START_UTC = new Date(startMillis).format("yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone.getTimeZone('UTC'))
                    env.QUEUED_DURATION_SEC = ((startMillis - queuedMillis) / 1000).toString()
                    env.RUN_ID = "${env.BUILD_TAG}-${startMillis}"
                }
                sh '''#!/usr/bin/env bash
                set -euo pipefail
                host="$(hostname)"
                total_mem_gb=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
                free_mem_gb=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
                used_mem_gb="0.0"
                cat > "${LAST_MEM_JSON}" <<JSON
{"run_id":"${RUN_ID}","build_number":"${BUILD_NUMBER}","analysis_name":"${ANALYSIS_NAME}","project_id":"${PROJECT_ID}","astree_version":"${ASTREE_VERSION}","config_profile":"${CONFIG_PROFILE}","host":"${host}","status":"QUEUED","start_time_utc":"${BUILD_START_UTC}","end_time_utc":"","duration_sec":0,"queued_time_utc":"${QUEUED_TIME_UTC}","queued_duration_sec":${QUEUED_DURATION_SEC},"used_mem_gb":${used_mem_gb},"server_free_mem_gb":${free_mem_gb},"total_mem_gb":${total_mem_gb},"repo_url":"${REPO_URL}","branch":"${BRANCH}"}
JSON
                cat "${LAST_MEM_JSON}"
                '''
                emailext(
                    to: params.EMAIL_TO,
                    subject: "[Astrée] STARTED #${env.BUILD_NUMBER} - ${params.ANALYSIS_NAME}",
                    body: """
Build started.

Build: ${env.BUILD_NUMBER}
Analysis: ${params.ANALYSIS_NAME}
Project: ${params.PROJECT_ID}
Start (UTC): ${env.BUILD_START_UTC}
Queued (UTC): ${env.QUEUED_TIME_UTC}
Queued Duration (sec): ${env.QUEUED_DURATION_SEC}
URL: ${env.BUILD_URL}

Last Known Memory: ${env.WORKSPACE}/${env.LAST_MEM_JSON}
"""
                )
            }
        }

        stage('Checkout') {
            steps {
                git(
                    url: params.REPO_URL,
                    branch: params.BRANCH,
                    credentialsId: env.REPO_CREDENTIALS_ID
                )
            }
        }

        stage('Analysis') {
            steps {
                lock(resource: env.LOCK_RESOURCE, quantity: 1) {
                    script {
                        timeout(time: params.TIMEOUT_HOURS.toInteger(), unit: 'HOURS') {
                            sh '''#!/usr/bin/env bash
                            set -euo pipefail

                            host="$(hostname)"
                            heartbeat_sec=$((HEARTBEAT_MIN * 60))
                            safe_min_free_gb="${SAFE_MIN_FREE_GB}"
                            total_mem_gb=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)

                            json_line() {
                                local status="$1"
                                local end_time_utc="$2"
                                local duration_sec="$3"
                                local used_mem_gb="$4"
                                local free_mem_gb="$5"
                                cat <<JSON
{"run_id":"${RUN_ID}","build_number":"${BUILD_NUMBER}","analysis_name":"${ANALYSIS_NAME}","project_id":"${PROJECT_ID}","astree_version":"${ASTREE_VERSION}","config_profile":"${CONFIG_PROFILE}","host":"${host}","status":"${status}","start_time_utc":"${BUILD_START_UTC}","end_time_utc":"${end_time_utc}","duration_sec":${duration_sec},"queued_time_utc":"${QUEUED_TIME_UTC}","queued_duration_sec":${QUEUED_DURATION_SEC},"used_mem_gb":${used_mem_gb},"server_free_mem_gb":${free_mem_gb},"total_mem_gb":${total_mem_gb},"repo_url":"${REPO_URL}","branch":"${BRANCH}"}
JSON
                            }

                            get_used_mem_gb() {
                                local pid="$1"
                                if [[ -n "${pid}" && -r "/proc/${pid}/status" ]]; then
                                    awk '/VmRSS/ {printf "%.1f", $2/1024/1024}' "/proc/${pid}/status"
                                else
                                    echo "0.0"
                                fi
                            }

                            get_free_mem_gb() {
                                awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo
                            }

                            start_epoch=$(date +%s)
                            start_time_utc="${BUILD_START_UTC}"

                            if [[ "${DRY_RUN}" == "true" ]]; then
                                used_mem_gb="0.0"
                                free_mem_gb="$(get_free_mem_gb)"
                                duration_sec=0
                                json_line "START" "" "${duration_sec}" "${used_mem_gb}" "${free_mem_gb}" | tee "${SUMMARY_JSON}"
                                echo "DRY_RUN enabled; skipping analysis command."
                                json_line "COMPLETED" "$(date -u +%FT%TZ)" "${duration_sec}" "${used_mem_gb}" "${free_mem_gb}" | tee -a "${SUMMARY_JSON}"
                                exit 0
                            fi

                            json_line "START" "" 0 "0.0" "$(get_free_mem_gb)" | tee "${SUMMARY_JSON}"

                            # Placeholder: replace with actual Astrée command/script.
                            ./run_astree.sh \
                                --project "${PROJECT_ID}" \
                                --profile "${CONFIG_PROFILE}" \
                                --analysis "${ANALYSIS_NAME}" \
                                --version "${ASTREE_VERSION}" \
                                --max-mem-gb "${MAX_MEM_GB}" \
                                --kloc "${KLOC}" &
                            astree_pid=$!
                            echo "${astree_pid}" > astree.pid

                            while kill -0 "${astree_pid}" >/dev/null 2>&1; do
                                sleep "${heartbeat_sec}"
                                used_mem_gb="$(get_used_mem_gb "${astree_pid}")"
                                free_mem_gb="$(get_free_mem_gb)"
                                duration_sec=$(( $(date +%s) - start_epoch ))
                                json_line "RUNNING" "" "${duration_sec}" "${used_mem_gb}" "${free_mem_gb}" | tee "${LAST_MEM_JSON}"

                                free_mem_int=${free_mem_gb%.*}
                                if (( free_mem_int < safe_min_free_gb )); then
                                    echo "Free memory below threshold (${safe_min_free_gb} GB). Terminating analysis." >&2
                                    kill -TERM "${astree_pid}" || true
                                    sleep 5
                                    kill -KILL "${astree_pid}" || true
                                    wait "${astree_pid}" || true
                                    end_time_utc="$(date -u +%FT%TZ)"
                                    duration_sec=$(( $(date +%s) - start_epoch ))
                                    json_line "FAILED" "${end_time_utc}" "${duration_sec}" "${used_mem_gb}" "${free_mem_gb}" | tee -a "${SUMMARY_JSON}"
                                    exit 1
                                fi
                            done

                            wait "${astree_pid}"
                            exit_code=$?
                            end_time_utc="$(date -u +%FT%TZ)"
                            duration_sec=$(( $(date +%s) - start_epoch ))
                            used_mem_gb="$(get_used_mem_gb "${astree_pid}")"
                            free_mem_gb="$(get_free_mem_gb)"

                            if [[ "${exit_code}" -eq 0 ]]; then
                                json_line "COMPLETED" "${end_time_utc}" "${duration_sec}" "${used_mem_gb}" "${free_mem_gb}" | tee -a "${SUMMARY_JSON}"
                            else
                                json_line "FAILED" "${end_time_utc}" "${duration_sec}" "${used_mem_gb}" "${free_mem_gb}" | tee -a "${SUMMARY_JSON}"
                                exit "${exit_code}"
                            fi
                            '''
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            script {
                def memStats = fileExists(env.LAST_MEM_JSON) ? readFile(env.LAST_MEM_JSON) : 'N/A'
                emailext(
                    to: params.EMAIL_TO,
                    subject: "[Astrée] SUCCESS #${env.BUILD_NUMBER} - ${params.ANALYSIS_NAME}",
                    body: """
Build succeeded.

Build: ${env.BUILD_NUMBER}
Analysis: ${params.ANALYSIS_NAME}
Start (UTC): ${env.BUILD_START_UTC}
End (UTC): ${new Date().format("yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone.getTimeZone('UTC'))}
URL: ${env.BUILD_URL}

Last Known Memory:
${memStats}
"""
                )
            }
        }
        failure {
            script {
                def memStats = fileExists(env.LAST_MEM_JSON) ? readFile(env.LAST_MEM_JSON) : 'N/A'
                emailext(
                    to: params.EMAIL_TO,
                    subject: "[Astrée] FAILED #${env.BUILD_NUMBER} - ${params.ANALYSIS_NAME}",
                    body: """
Build failed.

Build: ${env.BUILD_NUMBER}
Analysis: ${params.ANALYSIS_NAME}
Start (UTC): ${env.BUILD_START_UTC}
End (UTC): ${new Date().format("yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone.getTimeZone('UTC'))}
URL: ${env.BUILD_URL}

Last Known Memory:
${memStats}
"""
                )
            }
        }
        aborted {
            script {
                def memStats = fileExists(env.LAST_MEM_JSON) ? readFile(env.LAST_MEM_JSON) : 'N/A'
                emailext(
                    to: params.EMAIL_TO,
                    subject: "[Astrée] ABORTED #${env.BUILD_NUMBER} - ${params.ANALYSIS_NAME}",
                    body: """
Build aborted or timed out.

Build: ${env.BUILD_NUMBER}
Analysis: ${params.ANALYSIS_NAME}
Start (UTC): ${env.BUILD_START_UTC}
End (UTC): ${new Date().format("yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone.getTimeZone('UTC'))}
URL: ${env.BUILD_URL}

Last Known Memory:
${memStats}
"""
                )
            }
        }
        always {
            sh '''#!/usr/bin/env bash
            set -euo pipefail
            if [[ -f astree.pid ]]; then
                pid=$(cat astree.pid)
                if kill -0 "${pid}" >/dev/null 2>&1; then
                    kill -TERM "${pid}" || true
                    sleep 5
                    kill -KILL "${pid}" || true
                fi
            fi
            '''
            archiveArtifacts artifacts: "${SUMMARY_JSON},${LAST_MEM_JSON}", fingerprint: true, allowEmptyArchive: true
        }
    }
}
