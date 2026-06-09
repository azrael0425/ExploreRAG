$ErrorActionPreference = "Stop"

$taskGitCommit = (git rev-parse HEAD).Trim()
$taskGitBranch = (git branch --show-current).Trim()
$taskGitStatus = git status --porcelain
$taskGitDirty = if ($taskGitStatus) { "true" } else { "false" }

$env:EXPLORERAG_BUILD_GIT_COMMIT = $taskGitCommit
$env:EXPLORERAG_BUILD_GIT_BRANCH = $taskGitBranch
$env:EXPLORERAG_BUILD_GIT_DIRTY = $taskGitDirty

Write-Host "Building commit=$taskGitCommit branch=$taskGitBranch dirty=$taskGitDirty"
docker compose build backend frontend
if ($LASTEXITCODE -ne 0) {
    throw "docker compose build failed with exit code $LASTEXITCODE"
}
