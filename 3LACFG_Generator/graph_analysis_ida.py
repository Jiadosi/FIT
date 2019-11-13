from idautils import *
from idaapi import *
from idc import *

def getfunc_consts(func):
	strings = []
	consts = []
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	for bl in blocks:
		strs, conts = getBBconsts(bl)
		strings += strs
		consts += conts
	return strings, consts

def getConst(ea, offset):
	strings = []
	consts = []
	optype1 = GetOpType(ea, offset)
	if optype1 == o_imm:
		imm_value = GetOperandValue(ea, offset)
		if 0<= imm_value <= 10:
			consts.append(imm_value)
		else:
			if isLoaded(imm_value) and getseg(imm_value):
				str_value = GetString(imm_value)
				if str_value is None:
					str_value = GetString(imm_value+0x40000)
					if str_value is None:
						consts.append(imm_value)
					else:
						re = all(40 <= ord(c) < 128 for c in str_value)
						if re:
							strings.append(str_value)
						else:
							consts.append(imm_value)
				else:
					re = all(40 <= ord(c) < 128 for c in str_value)
					if re:
						strings.append(str_value)
					else:
						consts.append(imm_value)
			else:
				consts.append(imm_value)
	return strings, consts

def getBBconsts(bl):
	strings = []
	consts = []
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = GetMnem(inst_addr)
		if opcode in ['la','jalr','call', 'jal']:
			inst_addr = NextHead(inst_addr)
			continue
		strings_src, consts_src = getConst(inst_addr, 0)
		strings_dst, consts_dst = getConst(inst_addr, 1)
		strings += strings_src
		strings += strings_dst
		consts += consts_src
		consts += consts_dst
		try:
			strings_dst, consts_dst = getConst(inst_addr, 2)
			consts += consts_dst
			strings += strings_dst
		except:
			pass

		inst_addr = NextHead(inst_addr)
	return strings, consts

def getFuncCalls(func):
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calCalls(bl)
		sumcalls += callnum
	return sumcalls

def getLogicInsts(func):
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calLogicInstructions(bl)
		sumcalls += callnum
	return sumcalls

def getTransferInsts(func):
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calTransferIns(bl)
		sumcalls += callnum
	return sumcalls

def getArithmeticInsts(func):  # dosi @11.4
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calArithmeticIns(bl)
		sumcalls += callnum
	return sumcalls

def getIntrs(func):
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calInsts(bl)
		sumcalls += callnum
	return sumcalls	

def getLocalVariables(func):
	args_num = get_stackVariables(func.startEA)
	return args_num

def getGlobalVariables(func):  # dosi @ 11.1
	args_num = get_DMRVariables(func)
	return args_num

def getBasicBlocks(func):
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	return len(blocks)

def getIncommingCalls(func):
	refs = CodeRefsTo(func.startEA, 0)
	re = len([v for v in refs])
	return re


def get_stackVariables(func_addr):
    #print func_addr
    args = []
    stack = GetFrame(func_addr)
    if not stack:
            return 0
    firstM = GetFirstMember(stack)
    lastM = GetLastMember(stack)
    i = firstM
    while i <=lastM:
        mName = GetMemberName(stack,i)
        mSize = GetMemberSize(stack,i)
        if mSize:
                i = i + mSize
        else:
                i = i+4
        if mName not in args and mName and 'var_' in mName:
            args.append(mName)
    return len(args)

def get_DMRVariables(func):  # dosi @ 11.1
	#Direct memory reference = global variable
	args = []
	blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
	for bl in blocks:
		# print bl
		start = bl[0]
		end = bl[1]
		inst_addr = start
		while inst_addr < end:
			# opcode = GetMnem(inst_addr)
			op0 = GetOpType(inst_addr, 0)
			op0_val = GetOperandValue(inst_addr, 0)
			op1 = GetOpType(inst_addr, 1)
			op1_val = GetOperandValue(inst_addr, 1)
			# print opcode, op0, op0_val, op1, op1_val
			inst_addr = NextHead(inst_addr)
			if op0_val not in args and op0_val and op0 == 2:
				args.append(op0_val)
			if op1_val not in args and op1_val and op1 == 2:
				args.append(op1_val)
			# print args
	return len(args)

def calArithmeticIns(bl):
	x86_AI = {'add':1, 'sub':1, 'div':1, 'imul':1, 'idiv':1, 'mul':1, 'shl':1, 'dec':1, 'inc':1}
	mips_AI = {'add':1, 'addu':1, 'addi':1, 'addiu':1, 'mult':1, 'multu':1, 'div':1, 'divu':1, 'sub':1, 'subu':1, 'mfhi':1, 'mflo':1}  # dosi @11.4
 	arm_AI = {'add':1, 'adc':1, 'sub':1, 'sbc':1, 'mul':1, 'mla':1, 'umull':1, 'umlal':1, 'smull':1, 'smlal':1}  # dosi @11.4
	calls = {}
	calls.update(x86_AI)
	calls.update(mips_AI)
	calls.update(arm_AI)  # dosi @11.4
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = GetMnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = NextHead(inst_addr)
	return invoke_num

def calCalls(bl):
	calls = {'call':1, 'jal':1, 'jalr':1}
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = GetMnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = NextHead(inst_addr)
	return invoke_num

def calInsts(bl):
	start = bl[0]
	end = bl[1]
	ea = start
	num = 0
	while ea < end:
		num += 1
		ea = NextHead(ea)
	return num

# dosi @11.11 
# return list of insts of a bb
def collectInsts(bl, arch):
	insts_list, preprocess_insts_list = [], []  # dosi @11.12 add preprocessing rules
	start = bl[0]
	end = bl[1]
	inst_addr = start
	while inst_addr < end:
		insts_list.append(idc.GetDisasm(inst_addr))
		print 'Disasm:', idc.GetDisasm(inst_addr)
		preprocess_insts_list.append(preprocessing_rules(inst_addr, arch))  # dosi @11.12
		inst_addr = NextHead(inst_addr)
	return insts_list, preprocess_insts_list  # dosi @11.12

# dosi @11.12
# preprocessing inst
def preprocessing_rules(inst_addr, arch):
	res = ''
	res += GetMnem(inst_addr)  # keep opcode unchange
	res += '~'
	print 'arch', arch
	for offset in [0, 1, 2]:
		try:
			opType = GetOpType(inst_addr, offset)
			print opType
			if opType == o_void:  # 0
				break
			if opType == o_far or opType == o_near:  # 6 7
				res += 'FOO,'
			elif opType == o_reg:  # 1
				res += GetOpnd(inst_addr, offset) + ','  # x86, arm
			elif opType == o_displ:  # 4
				if arch == 'ARM':
					tmp = GetOpnd(inst_addr, offset).split(',')
					print 'tmp: ', tmp
					if len(tmp) == 1:
						tmp = tmp[0][1:-1]
					else:
						if tmp[0][-1] == ']':
							tmp = tmp[0][1:-1]
						else:
							tmp = tmp[0][1:]
					res += '[{}+0]'.format(tmp) + ','
				elif arch == 'metapc':
					res += '[{}+0]'.format(GetOpnd(inst_addr, offset).split('+')[0][1:]) + ','  # x86
				elif arch == 'mipsb':
					res += '[{}+0]'.format(GetOpnd(inst_addr, offset).split('$')[-1][:-1]) + ','  # mips
			elif opType == o_idpspec1:
				if arch == 'ARM':
					print '9', GetOpnd(inst_addr, offset)
					res += GetOpnd(inst_addr, offset) + ','
				else:
					pass
			else:
				strings, consts = getConst(inst_addr, offset)  # 5
				if strings and not consts:
					res += '<STR>,'
				elif consts and not strings:
					res += '0,'
				else:
					res += '<TAG>,'
		except:
			pass
	res = res[:-1]  # remove the last comma
	print res
	return res
		

def calLogicInstructions(bl):
	x86_LI = {'and':1, 'andn':1, 'andnpd':1, 'andpd':1, 'andps':1, 'andnps':1, 'test':1, 'xor':1, 'xorpd':1, 'pslld':1}
	mips_LI = {'and':1, 'andi':1, 'or':1, 'ori':1, 'xor':1, 'nor':1, 'slt':1, 'slti':1, 'sltu':1, 'xori':1}  #dosi @11.4
 	arm_LI = {'and':1, 'orr':1, 'eor':1, 'bic':1, 'orn':1, 'tst':1, 'teq':1}  # dosi @11.4
	calls = {}
	calls.update(x86_LI)
	calls.update(mips_LI)
	calls.update(arm_LI)  # dosi @11.4
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = GetMnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = NextHead(inst_addr)
	return invoke_num

def calSconstants(bl):
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = GetMnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = NextHead(inst_addr)
	return invoke_num


def calNconstants(bl):
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		optype1 = GetOpType(inst_addr, 0)
		optype2 = GetOpType(inst_addr, 1)
		if optype1 == 5 or optype2 == 5:
			invoke_num += 1
		inst_addr = NextHead(inst_addr)
	return invoke_num

def retrieveExterns(bl, ea_externs):
	externs = []
	start = bl[0]
	end = bl[1]
	inst_addr = start
	while inst_addr < end:
		refs = CodeRefsFrom(inst_addr, 1)
		try:
			ea = [v for v in refs if v in ea_externs][0]
			externs.append(ea_externs[ea])
		except:
			pass
		inst_addr = NextHead(inst_addr)
	return externs

def calTransferIns(bl):
	x86_TI = {'jmp':1, 'jz':1, 'jnz':1, 'js':1, 'je':1, 'jne':1, 'jg':1, 'jle':1, 'jge':1, 'ja':1, 'jnc':1, 'call':1}
	mips_TI = {'beq':1, 'bne':1, 'bgtz':1, "bltz":1, "bgez":1, "blez":1, 'j':1, 'jal':1, 'jr':1, 'jalr':1}
	arm_TI = {'MVN':1, "MOV":1}
	calls = {}
	calls.update(x86_TI)
	calls.update(mips_TI)
	calls.update(arm_TI)  # dosi @11.4
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = GetMnem(inst_addr)
		re = [v for v in calls if opcode in v]
		if len(re) > 0:
			invoke_num += 1
		inst_addr = NextHead(inst_addr)
	return invoke_num