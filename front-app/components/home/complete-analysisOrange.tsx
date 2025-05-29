"use client"

import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, CheckCircle, AlertCircle, Calculator, TrendingUp, User } from "lucide-react"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Slider } from "@/components/ui/slider"
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

const formSchema = z.object({
  age: z.number().min(17).max(90),
  capital_gain: z.number().min(0).max(99999),
  capital_loss: z.number().min(0).max(4356),
  hours_per_week: z.number().min(1).max(99),

  education_level: z
    .enum([
      "education_assoc_acdm",
      "education_assoc_voc",
      "education_bachelors",
      "education_doctorate",
      "education_hs_grad",
      "education_masters",
      "education_prof_school",
    ])
    .optional(),

  marital_status: z
    .enum([
      "marital_status_married",
      "marital_status_never_married",
      "marital_status_separated",
      "marital_status_widowed",
    ])
    .optional(),

  occupation: z
    .enum([
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
      "occupation_armed_forces",
    ])
    .optional(),

  race: z.enum(["race_amer_indian_eskimo", "race_asian_pac_islander", "race_other", "race_white"]).optional(),

  relationship: z
    .enum([
      "relationship_husband",
      "relationship_not_in_family",
      "relationship_other_relative",
      "relationship_own_child",
      "relationship_unmarried",
      "relationship_wife",
    ])
    .optional(),

  workclass: z
    .enum([
      "workclass_govt_employees",
      "workclass_never_worked",
      "workclass_private",
      "workclass_self_employed",
      "workclass_without_pay",
    ])
    .optional(),

  sex: z.enum(["sex_female", "sex_male"]).optional(),
})

type FormSchema = z.infer<typeof formSchema>
type FormKeys = keyof FormSchema

export function CompleteAnalysisOrange() {
  const [open, setOpen] = useState(false)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

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
    },
  })

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setLoading(true)
    try {
      console.log("values", values)
      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      })
      const data = await response.json()
      console.log(data)
      setPrediction(data.prediction)
      setOpen(true)
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-amber-50 p-4">
      <div className="max-w-6xl mx-auto">
        <Card className="border-orange-200 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-orange-100 to-amber-100 border-b border-orange-200">
            <div className="flex items-center space-x-3">
              <Calculator className="h-8 w-8 text-orange-600" />
              <div>
                <CardTitle className="text-2xl text-orange-900">Complete Prediction Analysis</CardTitle>
                <CardDescription className="text-orange-700">
                  Fill in demographic information to predict income level
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-6">
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                {/* Numerical features section */}
                <div className="space-y-6">
                  <div className="flex items-center space-x-3 pb-4 border-b border-orange-200">
                    <TrendingUp className="h-6 w-6 text-orange-600" />
                    <h3 className="text-2xl font-bold text-orange-900">Numerical Features</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FormField
                      control={form.control}
                      name="age"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Age</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={17}
                                max={90}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 17 | Max: 90</p>
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Capital Gains</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={0}
                                max={99999}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 0 | Max: 99999</p>
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Capital Loss</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={0}
                                max={4356}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 0 | Max: 4356</p>
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Hours per Week</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={1}
                                max={99}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 1 | Max: 99</p>
                            </div>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                </div>

                {/* Categorical features section */}
                <div className="space-y-6">
                  <div className="flex items-center space-x-3 pb-4 border-b border-orange-200">
                    <User className="h-6 w-6 text-orange-600" />
                    <h3 className="text-2xl font-bold text-orange-900">Categorical Features</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <FormField
                      control={form.control}
                      name="education_level"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Education</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "education_assoc_acdm", label: "Associate Degree (Academic)" },
                                { value: "education_assoc_voc", label: "Associate Degree (Vocational)" },
                                { value: "education_bachelors", label: "Bachelor's Degree" },
                                { value: "education_doctorate", label: "Doctorate" },
                                { value: "education_hs_grad", label: "High School Graduate" },
                                { value: "education_masters", label: "Master's Degree" },
                                { value: "education_prof_school", label: "Professional School" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Marital Status</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "marital_status_married", label: "Married" },
                                { value: "marital_status_never_married", label: "Never Married" },
                                { value: "marital_status_separated", label: "Separated" },
                                { value: "marital_status_widowed", label: "Widowed" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200 md:col-span-2 lg:col-span-1">
                          <FormLabel className="text-orange-900 font-semibold">Occupation</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="grid grid-cols-1 gap-2 max-h-64 overflow-y-auto"
                            >
                              {[
                                { value: "occupation_adm_clerical", label: "Administrative/Clerical" },
                                { value: "occupation_craft_repair", label: "Craft/Repair" },
                                { value: "occupation_exec_managerial", label: "Executive/Managerial" },
                                { value: "occupation_farming_fishing", label: "Farming/Fishing" },
                                { value: "occupation_handlers_cleaners", label: "Handlers/Cleaners" },
                                { value: "occupation_machine_op_inspct", label: "Machine Operator/Inspector" },
                                { value: "occupation_priv_house_serv", label: "Private Household Service" },
                                { value: "occupation_prof_specialty", label: "Professional Specialty" },
                                { value: "occupation_protective_serv", label: "Protective Service" },
                                { value: "occupation_sales", label: "Sales" },
                                { value: "occupation_tech_support", label: "Technical Support" },
                                { value: "occupation_transport_moving", label: "Transportation/Moving" },
                                { value: "occupation_armed_forces", label: "Armed Forces" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800 text-sm">{option.label}</FormLabel>
                                </FormItem>
                              ))}
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Race</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "race_amer_indian_eskimo", label: "American Indian/Eskimo" },
                                { value: "race_asian_pac_islander", label: "Asian/Pacific Islander" },
                                { value: "race_other", label: "Other" },
                                { value: "race_white", label: "White" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Relationship</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "relationship_husband", label: "Husband" },
                                { value: "relationship_not_in_family", label: "Not in Family" },
                                { value: "relationship_other_relative", label: "Other Relative" },
                                { value: "relationship_own_child", label: "Own Child" },
                                { value: "relationship_unmarried", label: "Unmarried" },
                                { value: "relationship_wife", label: "Wife" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Work Class</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "workclass_govt_employees", label: "Government Employee" },
                                { value: "workclass_never_worked", label: "Never Worked" },
                                { value: "workclass_private", label: "Private Sector" },
                                { value: "workclass_self_employed", label: "Self Employed" },
                                { value: "workclass_without_pay", label: "Without Pay" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
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
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Sex</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "sex_female", label: "Female" },
                                { value: "sex_male", label: "Male" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                </div>

                <div className="flex justify-center pt-6">
                  <Button
                    type="submit"
                    disabled={loading}
                    className="bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-semibold px-8 py-3 text-lg shadow-lg"
                    onClick={() => {
                      const formValues = form.getValues()
                      const missingValues = Object.entries(formValues)
                        .filter(([_, value]) => value === undefined || value === 0)
                        .map(([key]) => key)
                      console.log("Missing values:", missingValues)
                    }}
                  >
                    {loading && <Loader2 className="mr-2 h-5 w-5 animate-spin" />}
                    {loading ? "Analysis in progress..." : "Run Prediction"}
                  </Button>
                </div>
              </form>
            </Form>

            <AlertDialog open={open} onOpenChange={setOpen}>
              <AlertDialogContent className="border-orange-200">
                <AlertDialogHeader>
                  <AlertDialogTitle className="text-orange-900 flex items-center space-x-2">
                    {prediction === ">50K" ? (
                      <CheckCircle className="h-6 w-6 text-green-500" />
                    ) : (
                      <AlertCircle className="h-6 w-6 text-orange-500" />
                    )}
                    <span>Prediction Result</span>
                  </AlertDialogTitle>
                  <AlertDialogDescription asChild>
                    <div className="text-lg">
                      {prediction ? (
                        <div className="space-y-3">
                          <div className="text-center">
                            <div
                              className={`text-2xl font-bold p-4 rounded-lg ${
                                prediction === ">50K"
                                  ? "bg-green-100 text-green-800 border border-green-200"
                                  : "bg-orange-100 text-orange-800 border border-orange-200"
                              }`}
                            >
                              Predicted income: {prediction}
                            </div>
                          </div>
                          <p className="text-orange-700">
                            The model predicts that this person will have an income{" "}
                            <strong>
                              {prediction === ">50K" ? "greater than $50,000" : "less than or equal to $50,000"}
                            </strong>{" "}
                            per year.
                          </p>
                        </div>
                      ) : (
                        <div className="text-center text-orange-600">No prediction available.</div>
                      )}
                    </div>
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel
                    onClick={() => setOpen(false)}
                    className="border-orange-300 text-orange-700 hover:bg-orange-50"
                  >
                    Close
                  </AlertDialogCancel>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
