$TaskName = "DGMF_Direction_ID_Gate_20260809"
$ScriptPath = "D:\code\duomotai\chem5090\DGMF_repro_20260808\scripts\run_direction_id_gate_5090.ps1"
$TaskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`""

schtasks.exe /Delete /TN $TaskName /F 2>$null | Out-Null
schtasks.exe /Create /TN $TaskName /TR $TaskCommand /SC ONCE /ST 23:59 /RL HIGHEST /F
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
schtasks.exe /Run /TN $TaskName
exit $LASTEXITCODE
