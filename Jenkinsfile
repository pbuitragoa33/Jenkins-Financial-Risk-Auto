pipeline {
    agent any

    environment {
        GITHUB_EVENT_TYPE = 'push'
    }

    stages {

        stage('Detectar evento GitHub') {
            steps {
                script {
                    if (env.CHANGE_ID) {
                        env.GITHUB_EVENT_TYPE = 'pull_request'
                        echo "Evento detectado: Pull Request #${env.CHANGE_ID} (${env.CHANGE_TARGET} <- ${env.CHANGE_BRANCH})"
                    } else {
                        def commitMsg = sh(
                            script: "cd /workspace && git log -1 --pretty=%s",
                            returnStdout: true
                        ).trim()

                        if (env.BRANCH_NAME == 'main' && commitMsg.toLowerCase().contains('merge')) {
                            env.GITHUB_EVENT_TYPE = 'merge'
                            echo "Evento detectado: Merge a main (${commitMsg})"
                        } else if (currentBuild.previousBuild == null) {
                            env.GITHUB_EVENT_TYPE = 'branch_created'
                            echo "Evento detectado: Creacion de rama (${env.BRANCH_NAME})"
                        } else {
                            env.GITHUB_EVENT_TYPE = 'push'
                            echo "Evento detectado: Push en rama ${env.BRANCH_NAME}"
                        }
                    }
                }
            }
        }

        stage('Validar proyecto (pyops)') {
            steps {
                sh '''
                    cd /workspace
                    python pyops/validador_proyecto.py --verbose
                '''
            }
        }

        stage('Validar secrets (Gitleaks)') {
            steps {
                sh '''
                    cd /workspace
                    gitleaks detect --source . --no-git
                '''
            }
        }

        stage('Validar docker-compose') {
            steps {
                sh '''
                    cd /workspace
                    if [ ! -s docker-compose-jenkins.yml ]; then
                        echo "docker-compose vacío o no encontrado"
                        exit 1
                    fi
                '''
            }
        }

    }

    post {
        success {
            script {
                def resumen = "SUCCESS | evento=${env.GITHUB_EVENT_TYPE} | rama=${env.BRANCH_NAME ?: 'N/A'}"
                currentBuild.description = resumen
                echo "NOTIFICACION JENKINS: ${resumen}"
            }
        }
        failure {
            script {
                def resumen = "FAILURE | evento=${env.GITHUB_EVENT_TYPE} | rama=${env.BRANCH_NAME ?: 'N/A'}"
                currentBuild.description = resumen
                echo "NOTIFICACION JENKINS: ${resumen}"
            }
        }
        unstable {
            script {
                def resumen = "UNSTABLE | evento=${env.GITHUB_EVENT_TYPE} | rama=${env.BRANCH_NAME ?: 'N/A'}"
                currentBuild.description = resumen
                echo "NOTIFICACION JENKINS: ${resumen}"
            }
        }
        aborted {
            script {
                def resumen = "ABORTED | evento=${env.GITHUB_EVENT_TYPE} | rama=${env.BRANCH_NAME ?: 'N/A'}"
                currentBuild.description = resumen
                echo "NOTIFICACION JENKINS: ${resumen}"
            }
        }
    }
}