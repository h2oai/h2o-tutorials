# Qwiklab Creator How-To

This document describes the steps for creating a lab and productionizing it.

We worked with a consulting firm called Tudip to complete this work.  Dipti Agrawal was the contact there [ dipti at tudip dot com ].


---

## Creating your Lab

* Create an account at h2oai.qwiklab.com.

* You need to have "Creator" privilege in Qwiklab to make new labs.

* The "Creator^s" privilege is a super creator, who can access all labs.

### 1.  Add a new lab

* Click the + at the top right to create a new lab.

#### Lab Setup tab

* Lab Type:  Student

* Cloud Environment:  Amazon Web Service

* Lab Title:  [your title]

* Lab Version:  1

* Lab Description:  [your description]

* Lab Icon File:  You MUST provide one.

* Access Time (min):  57 and 117 minutes are good choices.  Keep it just under a rounded hour mark to avoid getting billed for an additional wasted hour.

#### Fleet Setup tab

(Note:  Cfn stands for CloudFormation)

* Default AWS Region:  us-west-2

* Valid AWS Regions:  us-west-2

* Cfn template document.  Here is one you can start from.  It contains:

  * Instance type (e.g. m4.4xlarge)
  * ami-id (e.g. ami-b4548fcc)
  * Ports to open (FromPort, ToPort) in the Security Group
  * Outputs of different URLs to show the student (e.g. "Driverless AI URL")

```
{
  "AWSTemplateFormatVersion": "2010-09-09",

  "Description": "CloudFormation",

  "Parameters": {
    "InstanceType": {
      "Description": "EC2 instance type",
      "Type" : "String",
      "Default" : "m4.4xlarge",
      "NoEcho": "true",
      "AllowedValues" : [ "m4.4xlarge"],
      "ConstraintDescription" : "must be a valid EC2 instance type."
    },
    "KeyName":{
      "Description":"Name of an existing EC2 KeyPair to enable access to qwiklabInstance",
      "Default":"generic-qwiklab",
      "Type":"String"
    },
    "AdministratorUser": {
      "Description": "Username the student logs into",
      "Default": "ubuntu",
      "Type": "String"
    }
  },

  "Resources": {
    "qwiklabInstance": {
      "Type": "AWS::EC2::Instance",
      "Properties": {
        "ImageId" : "ami-b4548fcc",
        "InstanceType": { "Ref": "InstanceType" },
        "SecurityGroups": [ { "Ref": "qwiklabInstanceSecurityGroup" } ],
        "KeyName": { "Ref": "KeyName" },
        "UserData": {
          "Fn::Base64": {
            "Fn::Join": ["", [
              "#!/bin/bash -x\n",
              "/bin/echo userdata_end > /tmp/userdata_end\n"
            ]]
          }
        }
      }
    },

    "qwiklabInstanceSecurityGroup": {
      "Type": "AWS::EC2::SecurityGroup",
      "Properties": {
        "GroupDescription": "Enable ports",
        "SecurityGroupIngress": [
          {
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "CidrIp": "0.0.0.0/0"
          },
          {
            "IpProtocol": "tcp",
            "FromPort": 80,
            "ToPort": 80,
            "CidrIp": "0.0.0.0/0"
          },
          {
            "IpProtocol": "tcp",
            "FromPort": 443,
            "ToPort": 443,
            "CidrIp": "0.0.0.0/0"
          },
          {
            "IpProtocol": "tcp",
            "FromPort": 8787,
            "ToPort": 8787,
            "CidrIp": "0.0.0.0/0"
          },
          {
            "IpProtocol": "tcp",
            "FromPort": 8888,
            "ToPort": 8888,
            "CidrIp": "0.0.0.0/0"
          },
          {
            "IpProtocol": "tcp",
            "FromPort": 12345,
            "ToPort": 12345,
            "CidrIp": "0.0.0.0/0"
          },
          {
            "IpProtocol": "tcp",
            "FromPort": 54321,
            "ToPort": 54321,
            "CidrIp": "0.0.0.0/0"
          }
        ]
      }
    }
  },

  "Outputs": {
    "DriverlessAI": {
      "Description": "Driverless AI URL",
      "Value": { "Fn::Join": [ "", ["http://", { "Fn::GetAtt": ["qwiklabInstance", "PublicIp"] }]]}
    }
  }
}
```

* AWS Instance Type:  Match the instance name in the cfn template.

#### Student Display tab

* Display lab connection buttons for:  Select "Custom".

#### Instructions tab

Add a .md file with any instructions you want to display.

### 2.  First test of your lab

In the lab detail view, you can click "Start" to launch it right there.  This is not what Students will see though.

### 3.  Add a new course

Click the + at the top right to create a new course.

* Name:  [your name]

So far it has been convenient to have exactly one lab per course.  So just copying the lab title has worked well.

* Version:  1

* Description:  [your description]

* Enabled:  true

* In the Lab section, check the box for the Lab to enable for this course.

### 4.  Add a new classroom

Click the + at the to right to create a new classroom.

* Connect the right course to your classroom.

* Choose whether the classroom should be Open (anyone can choose it) or restricted by a token (access code) or credits (i.e. for purchase by the student).  A credit costs $1.

* Add a filter to enable students access to the classroom.  A filter of `*` enables everybody.

---

## Productionizing your Lab

1.  Set the "Admin->Site Settings->HotLab duration" to a generous number.  40 minutes is a good number.

1.  A good strategy is to hide the class up front and reveal it at exactly the time people should start the lab.

1.  You can do this in the classroom view by clicking "Deactivate Lab".

1.  About 20 minutes before the class starts start Hotlabs by going to the classroom view and clicking "Pre-warm Labs".

1.  In the "Admin->Lab Info" view, you will see the "Running Hot Labs" count grow.

1.  Watch for a growing "Labs with Errors" count.  If it grows a little bit it's OK.  EC2 isn't 100% reliable.  I have seen occasionaly flakiness with security group construction for example.  A few rare failures won't impact the overall class.

1.  When the "Running Hot Labs" hits the target number and the class is ready to start, reveal the class by going to the classroom view and clicking "Activate Lab".

1.  Now the lab will show up in the Catalog view for students, and people can start them.

---

## Limits to be Aware of

The default Qwiklab environment worked well up to 150 Driverless AI instances.  (p2.xlarge, 128 GB EBS per instance)

To start more than 150 instances, the following two AWS limits needed to be increased.  These must be changed by the Qwiklab team themselves in their AWS production environment.  These kinds of changes typically require days to take effect after a request.  So do your scalability test at least a week ahead of time.

1.  CloudFormation stack limit was 200 by default.

    Symptom is:
    
    ```
    STACK:Aws::CloudFormation::Errors::LimitExceededException: Limit for stack has been exceeded
    ```

1.  AWS EBS volume limit.  Appeared to be 20 TB by default (by trial and error).

    Symptom is:
    
    ```
    Client.VolumeLimitExceeded
    ```

## Best Practices

### 1.  Test that the exact type and number of labs that you want to start will work ahead of time

The Qwiklab team will ask you how many instances you want enabled.  But there are other limits that can cause failures, as described above.  So you really need to test exactly what you need to have working (especially for a large event with hundreds of people).  Here's how.

1.  Use the HotLab feature to do this.

1.  In "Admin->Site Settings" set the HotLab duration to something short, like 20 minutes.  EC2 charges by the minute and Qwiklab only charges the $2.99/hr when a student connects, so you can run a scalability test without spending too much.

1.  For the particular course, set the "Max Hotlabs Allowed" to whatever number of instances you want to test.

1.  In the classroom, click "Pre Warm Labs".  This will start the instances automatically.

1.  In "Admin->Lab Info" you will see the number of "Running Hot Labs" grow.

1.  Watch to see if the "Labs With Errors" number grows.

1.  Connect to one of the HotLabs as a student to make sure it's working.

1.  Let the HotLabs wind down naturally after the 20 minutes pass.  You don't need to kill them by hand.

---

## Additional Resources

### EC2 CloudFormation stack limits

* <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cloudformation-limits.html>

```
Stacks

Maximum number of AWS CloudFormation stacks that you can create.

200 stacks

To create more stacks, delete stacks that you don't need or request an increase in the maximum number of stacks in your AWS account. For more information, see AWS Service Limits in the AWS General Reference.
```
