' 无窗口启动 start_service.bat。
' 计划任务(ONLOGON)指向这个脚本,而不是直接指 bat —— 否则会留一个
' 可见终端窗口,一旦被手滑关掉,整个服务就跟着停了。
Dim shell, here
Set shell = CreateObject("WScript.Shell")
here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run """" & here & "\start_service.bat""", 0, False
