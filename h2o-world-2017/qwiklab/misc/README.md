This template probably doesn't work exactly as-is, but has sample code of how you might use cfn-signal to synchronously control the CloudFormation process.

You could have the UserData section run a little script in the VM that synchronously checks for all the services to be up before doing a cfn-signal.
