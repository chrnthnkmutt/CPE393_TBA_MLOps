import {
    Form,
    FormControl,
    FormDescription,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
  } from "@/components/ui/form"
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, CheckCircle, AlertCircle } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
// import { features } from "@/lib/features"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"

import { cn } from "@/lib/utils"
import { Slider } from "@/components/ui/slider"

const formSchema = z.object({

    age: z.number().min(17).max(90),
    capital_gain: z.number().min(0).max(99999),
    capital_loss: z.number().min(0).max(4356),
    hours_per_week: z.number().min(1).max(99),

    education_level: z.enum([
        "education_assoc_acdm",
        "education_assoc_voc",
        "education_bachelors",
        "education_doctorate",
        "education_hs_grad",
        "education_masters",
        "education_prof_school"
      ]).optional()	,

      marital_status: z.enum([
        "marital_status_married",
        "marital_status_never_married",
        "marital_status_separated",
        "marital_status_widowed"
      ]).optional(),
  
      occupation: z.enum([
        "occupation_adm_clerical",
        "occupation_craft_repair",
        "occupation_exec_managerial",
        "occupation_farming_fishing",
        "occupation_handlers_cleaners",
        "occupation_machine_op_inspct",
        "occupation_priv_house_serv",
        "occupation_prof_specialty",
        "occupation_protective_serv",
        "occupation_sales",
        "occupation_tech_support",
        "occupation_transport_moving",
        "occupation_armed_forces"
      ]).optional(),

      race: z.enum([
        "race_amer_indian_eskimo",
        "race_asian_pac_islander",
        "race_other",
        "race_white"
      ]).optional(),

      relationship: z.enum([
        "relationship_husband",
        "relationship_not_in_family",
        "relationship_other_relative",
        "relationship_own_child",
        "relationship_unmarried",
        "relationship_wife"
      ]).optional(),

      workclass: z.enum([
        "workclass_govt_employees",
        "workclass_never_worked",
        "workclass_private",
        "workclass_self_employed",
        "workclass_without_pay"
      ]).optional(),

      sex: z.enum([
        "sex_female",
        "sex_male"
      ]).optional(),


  })

type FormSchema = z.infer<typeof formSchema>;
type FormKeys = keyof FormSchema;


export function CompleteAnalysis() {
    
    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            age: 0,
            capital_gain: 0,
            capital_loss: 0,
            hours_per_week: 0,
            education_level: undefined,
            marital_status: undefined,
            occupation: undefined,
            race: undefined,
            relationship: undefined,
            workclass: undefined,
            sex: undefined,
            }
    })
    
    async function onSubmit(values: z.infer<typeof formSchema>) {
        try {
            const response = await fetch("http://localhost:5000/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(values)
            });
            const data = await response.json();
            console.log(data);
        } catch (error) {
            console.error("Error:", error);
        }
    }

    return (
        <Card className="m-4">
            <CardHeader>
                <CardTitle>Complete Analysis</CardTitle>
            </CardHeader>
            <CardContent>    
                <Form {...form}>
                    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                        <div className="flex flex-col gap-4">
                            <h3 className="text-2xl font-bold">Numerical Features</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <FormField
                                    control={form.control}
                                    name="age"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Age</FormLabel>
                                            <FormControl>
                                                <div className="space-y-2">
                                                    <Slider
                                                        min={17}
                                                        max={90}
                                                        step={1}
                                                        value={[field.value]}
                                                        onValueChange={(value) => field.onChange(value[0])}
                                                    />
                                                    <Input 
                                                        type="number"
                                                        value={field.value}
                                                        onChange={(e) => field.onChange(Number(e.target.value))}
                                                    />
                                                    <p className="text-sm text-muted-foreground">
                                                        Min: 17 | Max: 90
                                                    </p>
                                                </div>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="capital_gain"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Capital Gain</FormLabel>
                                            <FormControl>
                                                <div className="space-y-2">
                                                    <Slider
                                                        min={0}
                                                        max={99999}
                                                        step={1}
                                                        value={[field.value]}
                                                        onValueChange={(value) => field.onChange(value[0])}
                                                    />
                                                    <Input 
                                                        type="number"
                                                        value={field.value}
                                                        onChange={(e) => field.onChange(Number(e.target.value))}
                                                    />
                                                    <p className="text-sm text-muted-foreground">
                                                        Min: 0 | Max: 99999
                                                    </p>
                                                </div>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="capital_loss"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Capital Loss</FormLabel>
                                            <FormControl>
                                                <div className="space-y-2">
                                                    <Slider
                                                        min={0}
                                                        max={4356}
                                                        step={1}
                                                        value={[field.value]}
                                                        onValueChange={(value) => field.onChange(value[0])}
                                                    />
                                                    <Input 
                                                        type="number"
                                                        value={field.value}
                                                        onChange={(e) => field.onChange(Number(e.target.value))}
                                                    />
                                                    <p className="text-sm text-muted-foreground">
                                                        Min: 0 | Max: 4356
                                                    </p>
                                                </div>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="hours_per_week"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Hours per Week</FormLabel>
                                            <FormControl>
                                                <div className="space-y-2">
                                                    <Slider
                                                        min={1}
                                                        max={99}
                                                        step={1}
                                                        value={[field.value]}
                                                        onValueChange={(value) => field.onChange(value[0])}
                                                    />
                                                    <Input 
                                                        type="number"
                                                        value={field.value}
                                                        onChange={(e) => field.onChange(Number(e.target.value))}
                                                    />
                                                    <p className="text-sm text-muted-foreground">
                                                        Min: 1 | Max: 99
                                                    </p>
                                                </div>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                        </div>

                        <div className="flex flex-col gap-4">
                            <h3 className="text-2xl font-bold">Categorical Features</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 gap-y-8">
                                <FormField
                                    control={form.control}
                                    name="education_level"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Education</FormLabel>
                                            <FormControl>
                                                <RadioGroup
                                                    onValueChange={field.onChange}
                                                    defaultValue={field.value}
                                                    className="flex flex-col space-y-1"
                                                >
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="education_assoc_acdm" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Associate Degree (Academic)
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="education_assoc_voc" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Associate Degree (Vocational)
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="education_bachelors" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Bachelor's Degree
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="education_doctorate" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Doctorate
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="education_hs_grad" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            High School Graduate
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="education_masters" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Master's Degree
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="education_prof_school" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Professional School
                                                        </FormLabel>
                                                    </FormItem>
                                                </RadioGroup>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />

                                <FormField
                                    control={form.control}
                                    name="marital_status"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Marital Status</FormLabel>
                                            <FormControl>
                                                <RadioGroup
                                                    onValueChange={field.onChange}
                                                    defaultValue={field.value}
                                                    className="flex flex-col space-y-1"
                                                >
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="marital_status_married" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Married
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="marital_status_never_married" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Never Married
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="marital_status_separated" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Separated
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="marital_status_widowed" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Widowed
                                                        </FormLabel>
                                                    </FormItem>
                                                </RadioGroup>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />

                                <FormField
                                    control={form.control}
                                    name="occupation"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Occupation</FormLabel>
                                            <FormControl>
                                                <RadioGroup
                                                    onValueChange={field.onChange}
                                                    defaultValue={field.value}
                                                    className="grid grid-cols-2 gap-2"
                                                >
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_adm_clerical" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Administrative/Clerical
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_craft_repair" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Craft/Repair
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_exec_managerial" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Executive/Managerial
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_farming_fishing" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Farming/Fishing
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_handlers_cleaners" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Handlers/Cleaners
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_machine_op_inspct" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Machine Operator/Inspector
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_priv_house_serv" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Private Household Service
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_prof_specialty" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Professional Specialty
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_protective_serv" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Protective Service
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_sales" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Sales
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_tech_support" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Technical Support
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_transport_moving" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Transport/Moving
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="occupation_armed_forces" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Armed Forces
                                                        </FormLabel>
                                                    </FormItem>
                                                </RadioGroup>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />

                                <FormField
                                    control={form.control}
                                    name="race"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Race</FormLabel>
                                            <FormControl>
                                                <RadioGroup
                                                    onValueChange={field.onChange}
                                                    defaultValue={field.value}
                                                    className="flex flex-col space-y-1"
                                                >
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="race_amer_indian_eskimo" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            American Indian/Eskimo
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="race_asian_pac_islander" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Asian/Pacific Islander
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="race_other" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Other
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="race_white" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            White
                                                        </FormLabel>
                                                    </FormItem>
                                                </RadioGroup>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />

                                <FormField
                                    control={form.control}
                                    name="relationship"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Relationship</FormLabel>
                                            <FormControl>
                                                <RadioGroup
                                                    onValueChange={field.onChange}
                                                    defaultValue={field.value}
                                                    className="flex flex-col space-y-1"
                                                >
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="relationship_husband" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Husband
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="relationship_not_in_family" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Not in Family
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="relationship_other_relative" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Other Relative
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="relationship_own_child" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Own Child
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="relationship_unmarried" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Unmarried
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="relationship_wife" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Wife
                                                        </FormLabel>
                                                    </FormItem>
                                                </RadioGroup>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />

                                <FormField
                                    control={form.control}
                                    name="workclass"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Work Class</FormLabel>
                                            <FormControl>
                                                <RadioGroup
                                                    onValueChange={field.onChange}
                                                    defaultValue={field.value}
                                                    className="flex flex-col space-y-1"
                                                >
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="workclass_govt_employees" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Government Employee
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="workclass_never_worked" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Never Worked
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="workclass_private" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Private Sector
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="workclass_self_employed" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Self-Employed
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="workclass_without_pay" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Without Pay
                                                        </FormLabel>
                                                    </FormItem>
                                                </RadioGroup>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />

                                <FormField
                                    control={form.control}
                                    name="sex"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Sex</FormLabel>
                                            <FormControl>
                                                <RadioGroup
                                                    onValueChange={field.onChange}
                                                    defaultValue={field.value}
                                                    className="flex flex-col space-y-1"
                                                >
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="sex_female" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Female
                                                        </FormLabel>
                                                    </FormItem>
                                                    <FormItem className="flex items-center space-x-3 space-y-0">
                                                        <FormControl>
                                                            <RadioGroupItem value="sex_male" />
                                                        </FormControl>
                                                        <FormLabel className="font-normal">
                                                            Male
                                                        </FormLabel>
                                                    </FormItem>
                                                </RadioGroup>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                        </div>


                        <Button 
                            type="submit"
                            onClick={() => {
                                const formValues = form.getValues();
                                const missingValues = Object.entries(formValues)
                                    .filter(([_, value]) => value === undefined || value === 0)
                                    .map(([key]) => key);
                                console.log("Valeurs manquantes:", missingValues);
                            }}
                        >
                            Submit
                        </Button>
                    </form>
                </Form>
            </CardContent>
        </Card>
    )
}